# EmotionDetection_smartSelfie
## Introduction
This project aims to classify the emotion on a person's face into one of seven categories, using deep convolutional neural networks and click the picture of the person if he or she is happy. The model is trained on the *FER-2013 dataset* which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.
## Dependencies
* Python , [OpenCv](https://opencv.org/) , [Tensorflow](https://www.tensorflow.org/)
* To install dependencies *pip install -r requirements*

## Data Preparation
* The [original FER2013 dataset in Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) is available as a single csv file. I had converted into a dataset of images in the PNG format for training/testing and provided this as the dataset in the previous section.
