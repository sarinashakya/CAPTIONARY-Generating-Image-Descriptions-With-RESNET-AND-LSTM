from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================
# MODEL ARCHITECTURE (same as Colab)
# ============================================

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * features, axis=1)
        return context

class CNN_Encoder(tf.keras.Model):   # class CNN_Encoder(Model):
    def __init__(self, embed_dim):
        super().__init__()
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet'
        )
        base.trainable = False
        self.cnn = tf.keras.Model(base.input, base.output)
        self.fc = tf.keras.layers.Dense(embed_dim) # layers.Dense(embed_dim)

    def call(self, images):
        x = self.cnn(images)
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]
        
        x_reshaped = tf.reshape(x, (batch_size * height * width, channels))
        x_dense = self.fc(x_reshaped)
        x_final = tf.reshape(x_dense, (batch_size, height * width, self.fc.units))
        return x_final

class RNN_Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, units):
        super().__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(units)
        self.lstm = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, hidden, cell=None, training=True):
        if cell is None:
            cell = tf.zeros_like(hidden)

        x = self.embedding(x)
        if training:
            context = self.attention(features, hidden)
            context = tf.expand_dims(context, 1)
            context = tf.tile(context, [1, x.shape[1], 1])
        else:
            context = self.attention(features, hidden)
            context = tf.expand_dims(context, 1)

        x = tf.concat([context, x], axis=-1)
        output, h, c = self.lstm(x, initial_state=[hidden, cell])
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, h, c

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# ============================================
# LOAD TOKENIZER AND MODEL
# ============================================

print("Loading tokenizer...")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

VOCAB_SIZE = len(tokenizer.word_index) + 1
EMBED_DIM = 256
UNITS = 512

print("Building model...")
encoder = CNN_Encoder(EMBED_DIM)
decoder = RNN_Decoder(VOCAB_SIZE, EMBED_DIM, UNITS)

# Build models with dummy data
dummy_images = tf.random.uniform([1, 224, 224, 3])
_ = encoder(dummy_images)

dummy_input = tf.zeros([1, 1], dtype=tf.int32)
dummy_features = tf.random.uniform([1, 49, EMBED_DIM])
dummy_hidden = tf.zeros([1, UNITS])
dummy_cell = tf.zeros([1, UNITS])
_ = decoder(dummy_input, dummy_features, dummy_hidden, dummy_cell)

# Load checkpoint
print("Loading checkpoint...")
checkpoint_dir = 'coco_checkpoints'
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print(f"Restored from {ckpt_manager.latest_checkpoint}")
else:
    print("WARNING: No checkpoint found!")

print("Model loaded successfully!")

# ============================================
# CAPTION GENERATION FUNCTION
# ============================================

def generate_caption_beam_search(image, encoder, decoder, tokenizer, max_len=20, beam_width=3, repetition_penalty=1.2):
    """Generate caption using beam search - COMPLETE VERSION"""
    image = tf.expand_dims(image, 0)
    features = encoder(image)

    start_token = tokenizer.word_index.get("<start>")
    end_token = tokenizer.word_index.get("<end>")
    
    if start_token is None:
        raise ValueError("Start token not found in tokenizer")

    hidden = tf.zeros((1, decoder.units))
    cell = tf.zeros((1, decoder.units))

    sequences = [[[start_token], 0.0, hidden, cell]]

    for step in range(max_len):
        all_candidates = []

        for seq, score, hidden, cell in sequences:
            if seq[-1] == end_token:
                all_candidates.append([seq, score, hidden, cell])
                continue

            x = tf.constant([[seq[-1]]], dtype=tf.int32)
            x = decoder.embedding(x)

            context = decoder.attention(features, hidden)
            if len(context.shape) == 2:
                context = tf.expand_dims(context, 1)

            x_input = tf.concat([context, x], axis=-1)
            output, h, c = decoder.lstm(x_input, initial_state=[hidden, cell])
            output = tf.reshape(output, (-1, output.shape[2]))
            
            preds = decoder.fc(output)
            preds = tf.nn.softmax(preds, axis=-1)
            preds = tf.squeeze(preds, axis=0).numpy()
            
            # Apply repetition penalty
            for word_id in set(seq):
                if word_id < len(preds):
                    preds[word_id] /= repetition_penalty

            # ⭐ TEMPERATURE SAMPLING (ADD THIS) ⭐
            temperature = 0.7
            preds = preds ** (1.0 / temperature)
            preds = preds / preds.sum()

            top_ids = preds.argsort()[-beam_width:][::-1]

            for word_id in top_ids:
                candidate = [seq + [word_id], score - np.log(preds[word_id] + 1e-9), h, c]
                all_candidates.append(candidate)

        # Sort with length normalization
        sequences = sorted(all_candidates, key=lambda x: x[1] / len(x[0]))[:beam_width]

    # Pick best with length normalization
    completed = [s for s in sequences if s[0][-1] == end_token]
    if not completed:
        completed = sequences
    
    best_seq = min(completed, key=lambda x: x[1] / len(x[0]))[0]

    words = [tokenizer.index_word.get(id, "") for id in best_seq 
             if id not in [start_token, end_token]]
    
    return " ".join(words)

def load_and_preprocess_image(img_path):
    """Load and preprocess image for the model"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return tf.convert_to_tensor(img, dtype=tf.float32)

# ============================================
# FLASK ROUTES
# ============================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', caption="No file selected")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', caption="No file selected")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and preprocess image
            img_tensor = load_and_preprocess_image(filepath)
            
            # Generate caption
            caption = generate_caption_beam_search(
                img_tensor, encoder, decoder, tokenizer, 
                max_len=20, beam_width=3, repetition_penalty=1.2
            )
            
            file_path = f"/static/uploads/{filename}"
            return render_template('index.html', caption=caption, file_path=file_path)
        
        except Exception as e:
            print(f"Error generating caption: {e}")
            return render_template('index.html', caption=f"Error: {str(e)}")
    
    return render_template('index.html', caption="Invalid file type")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
