# CAPTIONARY-Generating-Image-Descriptions-With-RESNET-AND-LSTM

## Project Description

This project generates natural language descriptions for images using deep learning.
It combines a RESNET CNN for image feature extraction and an LSTM network for sequence generation to produce captions.

## Objective

To create a system that can automatically generate descriptive captions for images using deep learning techniques.

## Technology Used

- Python (NumPy, TensorFlow, Keras)

- RESNET

- LSTM

## Dataset

This project uses the **MS COCO 2014 dataset** for training and testing the image captioning model.
The dataset is not included in this repository due to its large size.

### Steps

1. Download the **MS COCO 2014 dataset** from the official website:
   https://cocodataset.org/#download

2. Download the following files:

   * Training images (train2014.zip)
   * Validation images (val2014.zip)
   * Captions annotations (captions_train-val2014.zip)

3. Extract the downloaded files.

4. Place the dataset folder in the root directory of the project.

5. After placing the dataset in the correct directory, run the project normally.

## How to Run

1. Clone the repository.

2. Install required libraries:

```bash
pip install -r requirements.txt
````

3. Download the MS COCO 2014 dataset and place it in the project directory.

4. Run the application:

```bash
python app.py
````

## Working Principle

1. Image features are extracted using RESNET.

2. Captions are tokenized and processed for sequence learning.

3. LSTM is trained on image features + caption sequences.

4. The trained model can generate captions for new images.

## About

- The full dataset can be downloaded from the link above.

- This project is suitable for learning image captioning with CNN+LSTM models.
