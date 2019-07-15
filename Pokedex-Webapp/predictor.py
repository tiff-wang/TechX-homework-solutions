from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

def preprocess_img(image):
    '''
	This method processes the image into a form that can be accepted by the model. 
	''' 
    image = cv2.resize(image, (96, 96))
    print(image.shape)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

class Predictor:

    def __init__(self):
        # load model, default graph, and class index
        self.model = load_model('model/pokedex.model')
        self.graph = tf.get_default_graph()
        self.lb = pickle.loads(open('model/lb.pickle', 'rb').read())

    def predict(self, request):
        '''
        This method reads the file uploaded from the Flask application POST request, 
        and performs a prediction using the trained model
        '''

        # load and preprocess image
        f = request.files['image'].read()
        image = cv2.imdecode(np.fromstring(f, np.uint8), cv2.IMREAD_COLOR)
        image = preprocess_img(image)

        # classification
        result = ''
        with self.graph.as_default():
            prob = self.model.predict(image)[0]
            index = np.argmax(prob)
            label = self.lb.classes_[index]
            result = label

        return result 