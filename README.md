# CAPTIONARY-Generating-Image-Descriptions-With-RESNET-AND-LSTM

## Project Description

This project generates natural language descriptions for images using deep learning.
It combines a RESNET CNN for image feature extraction and an LSTM network for sequence generation to produce captions.

## Objective

To create a system that can automatically generate descriptive captions for images using deep learning techniques.

## Technology Used

-Python (NumPy, TensorFlow, Keras)

-RESNET

-LSTM

## Dataset

This project uses the Flickr8k dataset.
Because the dataset is large, it is not included in this repository.

Download from:
[Flickr8k on Kaggle](https://www.kaggle.com/datasets/aladdinpersson/flickr8kimagescaptions)

After downloading:

-Extract images into dataset/Images/

-Extract captions file into dataset/captions

Only a few sample images are included here.

## How to Run

Install dependencies:

pip install -r requirements.txt

## Working Principle

-Image features are extracted using RESNET.

-Captions are tokenized and processed for sequence learning.

-LSTM is trained on image features + caption sequences.

-The trained model can generate captions for new images.

## About

-Sample images are included for demonstration purposes.

-The full dataset can be downloaded from the link above.

This project is suitable for learning image captioning with CNN+LSTM models.
