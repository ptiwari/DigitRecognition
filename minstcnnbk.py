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

def loadData():
	mnist = tf.keras.datasets.mnist
	(train_X,train_Y), (test_X,test_Y) = mnist.load_data()
	print(train_X.shape)
	return (train_X,train_Y), (test_X,test_Y)

def preProcessData(train_X,test_X):
	# Find the unique numbers from the train labels
	classes = np.unique(train_Y)
	nClasses = len(classes)
	train_X = train_X.reshape(-1, 28,28, 1)
	test_X = test_X.reshape(-1, 28,28, 1)
	train_X = train_X.astype('float32')
	test_X = test_X.astype('float32')
	train_X = train_X / 255.
	test_X = test_X / 255.
	# Change the labels from categorical to one-hot encoding
	train_Y_one_hot = kp.to_categorical(train_Y)
	test_Y_one_hot = kp.to_categorical(test_Y)

def splitData(train_X,)
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)


# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

batch_size = 64
epochs = 20
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

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.summary()

print(train_X.shape)
print(train_X[0,0,0,0])
print("Max:",np.amax(train_X))
fashion_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
model.save("mnist-model.h5")




