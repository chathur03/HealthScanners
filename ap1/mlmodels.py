import cv2
import imghdr
import os
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import torch
from ultralytics import YOLO
from PIL import Image, ImageOps     
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


# def predict_skinD(image_path):
#     model = tf.keras.models.load_model("static/models/skin_disease.h5")
#     arr = ['BA- cellulitis','BA-impetigo','FU-athlete-foot','FU-nail-fungus','FU-ringworm','PA-cutaneous-larva-migrans','VI-chickenpox','VI-shingles']
#     img = cv2.imread(image_path)
#     resize = tf.image.resize(img, (224,224))
#     yhat =  np.expand_dims(resize/255, 0)
#     print(resize.shape)
#     y_pred = model.predict(yhat)
#     print(y_pred)
#     pred = np.argmax(y_pred[0])
#     pred = arr[pred]
#     cv2.imwrite('static/pred/predicted.png', img)
#     return pred

def predict_lungD(image_path):
    model = tf.keras.models.load_model("static/models/lung_disease_predictor.h5")
    arr = ["Covid-19","Normal Lung","Pnuemonia","Tuberculosis"]
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

def predict_bone(image_path):
    model = YOLO("static/models/best.pt")
    results = model(image_path)
    results[0].save("static/pred/predicted.png")



    return ">"


def general_predict(image_path):
    np.set_printoptions(suppress=True)
    model = load_model("static/models/general_classifier.h5", compile=False)
    class_names = open("static/labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Confidence Score:", confidence_score)

    return class_name[2:]

def predict_skinD(image_path):
    np.set_printoptions(suppress=True)
    model = load_model("static/models/skin_disease.h5", compile=False)
    class_names = open("static/skin_labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    img = cv2.imread(image_path)
    resize = tf.image.resize(img, (224,224))
    cv2.imwrite('static/pred/predicted.png', img)

    print("Confidence Score:", confidence_score)

    return class_name[2:]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # It has a sequence of layers
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),  # First layer
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # Depth of feature maps increases with out_channels
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )

        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)  # Flattens the 2D array
        x = self.fc_model(x)
        x = torch.sigmoid(x)  # Use torch.sigmoid instead of F.sigmoid
        return x

model = CNN()
model = model.load_state_dict(torch.load('static/models/brain.pth', map_location='cpu'))


def predict_brain(image_path, model_path="static/models/brain.pth"):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)) 


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    device = torch.device('cpu')

    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  
    model.eval()  


    with torch.no_grad():  
        output = model(img_tensor)
        prediction = output.item()  

    img = cv2.imread(image_path)
    cv2.imwrite('static/pred/predicted.png', img)

    if prediction < 0.85:
        return "Gliomas Brain Tumor"
    else:
        return "Healthy Brain"