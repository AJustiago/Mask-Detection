from imutils.video import VideoStream
from keras.utils import img_to_array
from keras.models import load_model
from PIL import Image as im
import numpy as np
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

print("[INFO] starting video stream...")
vs = VideoStream(src=0, resolution= (1280, 720)).start()
time.sleep(2.0)

while True:
	frame_data = frame = vs.read()
	# frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
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
			img_face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			img_face = im.fromarray(img_face)
			img_face = img_face.resize((224,224))
			face = cv2.resize(face, (32, 32))
			keep_face = face
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]
			label = "{}: {:.4f}".format(label, preds[j])
			print(label)
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
	cv2.imshow("Frame", frame)
cv2.destroyAllWindows()
vs.stop()