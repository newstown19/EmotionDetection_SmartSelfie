# EmotionDetection_smartSelfie
## Introduction
This project aims to classify the emotion on a person's face into one of seven categories, using deep convolutional neural networks and click the picture of the person if he or she is happy. The model is trained on the **FER-2013 dataset** which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.
## Dependencies
* Python , [OpenCv](https://opencv.org/) , [Tensorflow](https://www.tensorflow.org/)
* To install dependencies *pip install -r requirements*

## Data Preparation
* The [original FER2013 dataset in Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) is available as a single csv file. I had converted into a dataset of images in the numpy array format for training/testing.
* In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. I have provided the code I wrote for data preprocessing in the FER_Preprocessing.ipynb file which can be used for reference.
## Folder Structure
* **Haarcascade** (folder) : contains haarcascade xml file for face detection
* **images_clicked** (folder) : will contain the clicked pictures by the camera when you smile :) .
* **model** (folder): contains CNN model structure's json file and it's respective weights.
* **running_image** (folder) : photos describing actual functinality, during run time of script real_time.py
* **FER_Preprocessing** (file) : preprocessing file for FER 2013 dataset.
* **VGG_2_Training** (file) : file for training CNN.
* **real_time** (file) : real-time prediction python file.

## Algorithm
* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to 48x48 and is passed as input to the CNN.

* The network outputs a list of softmax scores for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen along with probabilistic score of all possible output
* If score for label happy is prevalent for **10 frames** then, person picture is taken

## Example Output
![](/running_image/image1.png)


![](/running_image/Image2.png)

## References
* "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,
X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu, M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and Y. Bengio. arXiv 2013.
