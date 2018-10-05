import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras.utils as kp
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ReLU
from sklearn.model_selection import train_test_split

# Load the MNIST data from Keras API
def loadData():
	mnist = tf.keras.datasets.mnist
	(train_X,train_Y), (test_X,test_Y) = mnist.load_data()
	print(train_X.shape)
	return (train_X,train_Y), (test_X,test_Y)

# Reshape the image to the 28*28*1 size
# Convert the pixel value between 0 to 1
# Convert the integer data to float
# Convert the categorical data to one hot representation
def preProcessData(train_X,test_X,train_Y,test_Y):
	# Find the unique numbers from the train labels
	train_X = train_X.reshape(-1, 28,28, 1)
	test_X = test_X.reshape(-1, 28,28, 1)
	train_X = train_X.astype('float32')
	test_X = test_X.astype('float32')
	train_X = train_X / 255.
	test_X = test_X / 255.
	# Change the labels from categorical to one-hot encoding
	train_Y_one_hot = kp.to_categorical(train_Y)
	test_Y_one_hot = kp.to_categorical(test_Y)
	return train_X,test_X,train_Y_one_hot,test_Y_one_hot

# Split the training data into training and valid set
# Training data: 80% and Test data: 20%
def splitData(train_X,train_Y_one_hot):
	train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
	return train_X,valid_X,train_label,valid_label

# Create the model
# There are three convolution layer
# The filter is of size 3*3
# We used maxpooling of size 2*2
# We used zero padding to make the 
# activation map as the same size
# of input image
# We used two fully connected layer
# and soft max to output the class
def createModel():
	num_classes = 10
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
	model.add(ReLU())
	model.add(MaxPooling2D((2, 2),padding='same'))
	model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
	model.add(ReLU())
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Conv2D(512, (3, 3), activation='linear',padding='same'))
	model.add(ReLU())                  
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Flatten())
	model.add(Dense(1000, activation='linear'))
	model.add(ReLU())  
	model.add(Dense(500, activation='linear'))                
	model.add(Dense(num_classes, activation='softmax'))
	return model;

# Compiles the model
def compileModel(model):
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

# Train the model
def trainModel(model,train_X,train_label,valid_X,valid_label):
	batch_size = 64
	epochs = 20
	minst_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

# Test the model
def testModel(model,test_X,test_Y_one_hot):
	test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)

# Save the model
def save(model,fileName):
	model.save(fileName)

#Main method

def main():
	#Load data
	(train_X,train_Y), (test_X,test_Y) = loadData()
	#Preprocess the data
	train_X,test_X,train_Y_one_hot,test_Y_one_hot = preProcessData(train_X,test_X,train_Y,test_Y)
	#Split the data
	train_X,valid_X,train_label,valid_label = splitData(train_X,train_Y_one_hot);
	#Create Neural Network
	model = createModel();
	#Print the summary	
	model.summary()
	#Compile the model
	compileModel(model)
	#Train the model
	trainModel(model,train_X,train_label,valid_X,valid_label)
	#Test the model
	testModel(model,test_X,test_Y_one_hot)
	#save the model
	save(model,'mnist-model.h5')

if __name__== "__main__":
  main()



