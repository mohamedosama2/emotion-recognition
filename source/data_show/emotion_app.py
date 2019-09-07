# libraries
import cv2
import dlib
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from flask import Flask,render_template

# variables
emotion_dict={0:'Angry',
              1:'Disgust',
              2:'Fear',
              3:'Happy',
              4:'Sad',
              5:'Surprise',
              6:'Neutral'}
image="../../data/training/lato.jpg"
model_path="face_detector_model/mmod_human_face_detector.dat"

def image_ensuring(image):
    '''
    :param image:
    :ensuring that the input is an image and show it
    '''
    #image = secure_filename(image)
    l = image.split('.')
    if l[-1] in ["png", "jpg", "jpeg", "TIF", "GIF"]:
         return cv2.imread(image)
    else:
        exit()

def model_detector(image):
    '''
    :param image:
    :return:the image with rectangle of the the image after face detection
    '''
    cnnFaceDetector = dlib.cnn_face_detection_model_v1(model_path)
    faceRects = cnnFaceDetector(image, 1)
    for faceRect in faceRects:
        x1 = faceRect.rect.left()
        y1 = faceRect.rect.top()
        x2 = faceRect.rect.right()
        y2 = faceRect.rect.bottom()
    return image[y1:y2, x1:x2]


def image_processing(image):
    '''
    :return:
    :param image:
    :return:the emotion of image after reprocessing
    '''
     # convert image from RGB to Grayscal
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     # resizing the image to enter it in the model
    image =cv2.resize(image,(48,48), interpolation = cv2.INTER_AREA)
     #converting the pixels to float and multiply it to enter it  to the model
    image=image/255.0
     # reshaping the image
    image=np.array(image).reshape(1,48,48,1)
     # load the model of emotion recognition
    face_detector=load_model('../../emotion_model.h5py')
     # return the result of the image
    return emotion_dict[np.argmax(face_detector.predict(image))]

def application():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template("index.html")

    @app.route('/', methods=["POST"])
    def apply_value():
        root = tk.Tk
        root.withdraw
        file_path = filedialog.askopenfile()
        image = file_path.name
        image = image_ensuring(image)
        image = model_detector(image)
        emotion=image_processing(image)
        return render_template("pass.html", emotion=emotion)

    if __name__ == '__main__':
        app.run()

application()

