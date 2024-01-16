import matplotlib
matplotlib.use("Agg")

from MiniVGG.miniVGG_model import MiniVGG
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

INIT_LR = 1e-4
BS = 8
EPOCHS = 50

print("[INFO] loading images...")
imagePaths = list(paths.list_images('D:/Mask Detection/Dataset/data'))
data = []
labels = []

for imagePath in imagePaths:
    if "with_mask" in imagePath:
        label = "with_mask"
    else: label = "without_mask"
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    data.append(image)
    labels.append(label)
data = np.array(data, dtype="float") / 255.0

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, 
                         height_shift_range=0.2, shear_range=0.15, 
                         horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = MiniVGG.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY), 
              steps_per_epoch=len(trainX) // BS,epochs=EPOCHS)

print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] serializing network to '{}'...".format('D:/Mask Detection/MiniVGG'))
model.save("MaskDetection1.model", save_format="h5")
# save the label encoder to disk

f = open('D:/Mask Detection/md.pickle1', "wb")
f.write(pickle.dumps(le))
f.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('D:/Mask Detection/plot.png')
