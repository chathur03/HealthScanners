import cv2
import imghdr
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

def predict_skinD(image_path):
    model = tf.keras.models.load_model("static/models/skin_disease.h5")
    arr = ['BA- cellulitis','BA-impetigo','FU-athlete-foot','FU-nail-fungus','FU-ringworm','PA-cutaneous-larva-migrans','VI-chickenpox','VI-shingles']
    img = cv2.imread(image_path)
    resize = tf.image.resize(img, (224,224))
    yhat =  np.expand_dims(resize/255, 0)
    print(resize.shape)
    y_pred = model.predict(yhat)
    print(y_pred)
    pred = np.argmax(y_pred[0])
    pred = arr[pred]
    cv2.imwrite('static/pred/predicted.png', img)
    return pred

def predict_lungD(image_path):
    model = tf.keras.models.load_model("static/models/lung_disease_predictor.h5")
    arr = ["COVID19","NORMAL","PNEUMONIA","TUBERCULOSIS"]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=-1)
    resize = tf.image.resize(img, (256,256))
    yhat =  np.expand_dims(resize/255, 0)
    print(resize.shape)
    y_pred = model.predict(yhat)
    print(y_pred)
    pred = np.argmax(y_pred[0])
    pred = arr[pred-1]
    cv2.imwrite('static/pred/predicted.png', img)
    return pred