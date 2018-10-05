from keras.models import  load_model
from keras.preprocessing import image
import numpy as np
import sys

#Load Model file
def loadModel(modelFile):
	model = load_model(modelFile)
	return model;

def loadImage(imgFile):
	img = image.load_img(path=imgFile,grayscale=True,target_size=(28,28,1))
	img = image.img_to_array(img)
	img = img / 255.
	img = img.astype('float32')
	img = np.expand_dims(img,0);
	return img;

def predict(model,img):
	img_class = model.predict_classes(img)
	prob = model.predict(img);
	return prob,img_class

def main():
	
	if len(sys.argv)<2:
		print("No input image file specified. Using default img1.jpg.")
		imgFile = 'img1.jpg';
	else:
		imgFile = sys.argv[1];
	print("Predicting class for",imgFile);
	model = loadModel('mnist-model.h5')
	img = loadImage(imgFile)
	prob,img_class = predict(model,img);
	classname = img_class[0]
	print("Predicted numer is: ",classname," and probability is:",prob[0,classname])

if __name__== "__main__":
  main()
