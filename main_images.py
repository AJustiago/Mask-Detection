from imutils.video import VideoStream
from keras.utils import img_to_array
from keras.models import load_model
from PIL import Image as im
import numpy as np
from numpy import asarray
import imutils
import pickle
import time
import cv2

print("[INFO] loading face detector...")
protoPath = "D:/Mask Detection/face_detector/deploy.prototxt"
modelPath = "D:/Mask Detection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading mask detector...")
model = load_model(r"D:\mixed code\backend\mask\MaskDetection.model")
le = pickle.loads(open(r"D:\mixed code\backend\mask\md.pickle", "rb").read())

face = 'D:/Mask Detection/test.jpeg'
frame = im.open(face)
(h, w) = (600, 600)
print(frame.format)
print(frame.size)
print(frame.mode)
frame = asarray(frame)
frame = cv2.resize(frame, (640, 480))
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (640, 480)), 1.0, (640, 480), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.9:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)
        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (32, 32))
        
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        preds = model.predict(face)[0]
        j = np.argmax(preds)
        label = le.classes_[j]
        label = "{}: {:.4f}".format(label, preds[j])
        print(label)