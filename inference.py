from keras.models import  load_model
from keras.preprocessing import image
import numpy as np
import sys
import os
import tarfile
import matplotlib.pyplot as plt
 
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

#Check if mnist-model.h5 exists
# If not open the tar file
def extractModelFile(modelFile):
	exists = os.path.isfile('mnist-model.h5')
	if ~exists:
		try:
			tar = tarfile.open("mnist-model.h5.tar.gz")
			tar.extractall()
			tar.close()
		except:
			print("Can't Open tar file. Please traing the model")
	
# Display the image for 2 seconds
def showImage(fileName):
	img = image.load_img(fileName) # images are color images
	plt.gca().clear()
	plt.imshow(img);
	plt.draw()
	plt.pause(2)
	plt.close()

def main():
	
	if len(sys.argv)<2:
		print("No input image file specified. Using default img1.jpg.")
		imgFile = 'img_1.jpg';
	else:
		imgFile = sys.argv[1];
	print("Predicting class for",imgFile);
	modelFile = 'mnist-model.h5'
	extractModelFile(modelFile);
	model = loadModel(modelFile)
	img = loadImage(imgFile)
	prob,img_class = predict(model,img);
	classname = img_class[0]
	print("Predicted numer is: ",classname," and probability is:",prob[0,classname])
	print("The image is")
	showImage(imgFile)

if __name__== "__main__":
  main()
