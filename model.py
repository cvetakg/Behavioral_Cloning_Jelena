# SDCND Term1 Behavioral Cloning
# Author: Jelena Kocic
# Date: 17.04.2017.

import os
import pickle
import numpy as np
import csv
import cv2
import sys

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

import random
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from keras.optimizers import Adam

lines = []
with open('/Users/Jelena/CarND-Behavioral-Cloning-P3/mydata_small/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
		
images = []
measurements = []

NEW_IMAGES_FOLDER = '/Users/Jelena/CarND-Behavioral-Cloning-P3/mydata_small/IMG/'
def resize_image(image_file):
    image = plt.imread(NEW_IMAGES_FOLDER + image_file)
    return scipy.misc.imresize(image, (160, 320))

images = np.asarray([resize_image(image) for image in os.listdir('/Users/Jelena/CarND-Behavioral-Cloning-P3/mydata_small/IMG/')])
print (images.shape)

for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '/Users/Jelena/CarND-Behavioral-Cloning-P3/mydata_small/IMG' + filename
	speed = line[3]
	speed = speed.replace(",", ".")
	measurement = float(speed)
	measurements.append(measurement)
	

# argumenting images by flipping images horizontally
argumented_images, argumented_measurements = [], []
for image, measurement in zip(images, measurements):
	argumented_images.append(image)
	argumented_measurements.append(measurement)
	argumented_images.append(cv2.flip(image,1))
	argumented_measurements.append(measurement*-1.0)
	
# images and seering measurements	
X_train = np.array(argumented_images)
y_train = np.array(argumented_measurements)
print(X_train.shape)


# my_model
# The Sequential model is a linear stack of layers
model = Sequential()

# normalization -> data are now in range -0.5 to 0.5
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# cropping layer -> to remove sky, trees and part of the car on the bottom
model.add(Cropping2D(cropping=((70,25), (0,0))))

# NVIDIA network model
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
#model.add(Dropout(0.5))
model.add(Dense(1))

# mean square error = MSE (because this is regression network, instead of classification network)
# MSE to minimize difference between steering angle prediction and real steering angle
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)

# saving the model
model.save('model.h5')


