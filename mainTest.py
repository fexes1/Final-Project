import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score


model=load_model('BrainTumor10Epochs.h5')

image=cv2.imread('C:\\Users\\Nicholas\Videos\\BrainTumor Classification DL\\datasets\pred\\pred5.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)


input_img = np.expand_dims(img, axis=0)

result = (model.predict(input_img)>0.5).astype("int32")

# Display the prediction result
if result[0][0] == 0:
    prediction_label = 'No Tumor'
else:
    prediction_label = 'Tumor'

print(f"Predicted Label: {prediction_label}")



