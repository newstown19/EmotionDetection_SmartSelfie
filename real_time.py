from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import argparse
import cv2
import os
import json
import numpy as np

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("-sc","--scaleFactor" , help="Scale factor for Haarcascade classifier",dest = 'scale' , default = 1.1 , type =float)
    parser.add_argument("-mn","--minNeighbors" , help="MinNeighbors for Haarcascade classifier" ,dest = 'minN', default = 5 , type = int)

    ap = parser.parse_args()



    model_path = os.path.join(os.getcwd(),"model","Vgg_3_130.json")
    weights_path = os.path.join(os.getcwd(),"model","Vgg_3_130.hdf5")
    saveImage_path = os.path.join(os.getcwd() , "images_clicked")


    try:
        with open(model_path , 'r') as file:
            json_model = file.read()
            model = model_from_json(json_model)
        file.close()
        model.load_weights(weights_path)
        print("Model Loaded Sucessfully")
    except Exception as e:
        print("Could not load file : Error {} ".format(e))
    
    face_detection = cv2.CascadeClassifier(os.path.join(os.getcwd() , "Haarcascade" , "haarcascade_frontalface_default.xml"))


    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]
    counter = 0
    counter_stop = 20
    img_name = 0
    cv2.namedWindow('your_face')
    camera = cv2.VideoCapture(0)
    while True:
        frame = camera.read()[1]
        #reading the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=ap.scale,minNeighbors=ap.minN,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)


            preds = model.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            if label == 'happy':
                counter +=1
                if (counter >= counter_stop):
                    cv2.imwrite(os.path.join(saveImage_path , "Image" + str(img_name) +".png") , frame)
                    counter = 0
                    img_name +=1
                    print("Saving Picture...!")
        else: continue




        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    # draw the label + probability bar on the canvas
                   # emoji_face = feelings_faces[np.argmax(preds)]


                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)
    #    for c in range(0, 3):
    #        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
    #        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
    #        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
