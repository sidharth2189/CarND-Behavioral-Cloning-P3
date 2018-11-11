# import libraries
import csv
import cv2

import random
import numpy as np

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D

import sklearn
from sklearn.model_selection import train_test_split

# Read and store lines from driving data log
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skips header line in csv file that contains description
    for line in reader:
        lines.append(line)

# Correction value for steering angle for left and right camera images        
steering_correction = 0.2

# Create empty arrays to hold images and steering values        
images = []
steering_angles = []

# For each line in the driving data log, get camera image (left, right and centre) and steering value
for line in lines:
    # Read in images from center, left and right cameras
    image_centre = cv2.imread('data/'+ '/IMG/' + line[0].split('/')[-1])
    image_centre = cv2.cvtColor(image_centre, cv2.COLOR_BGR2RGB) # drive.py loads images in rgb
    image_left = cv2.imread('data/'+ '/IMG/' + line[1].split('/')[-1]) 
    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB) # drive.py loads images in rgb
    image_right = cv2.imread('data/'+ '/IMG/' + line[2].split('/')[-1])
    image_right  = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB) # drive.py loads images in rgb
    
    # create adjusted steering measurements for the side camera images
    steering_centre = float(line[3])
    steering_left = steering_centre + steering_correction
    steering_right = steering_centre - steering_correction
   
    # Add images and angles to data set
    images.extend([image_centre, image_left, image_right])
    steering_angles.extend([steering_centre, steering_left, steering_right])

cv2.imwrite('sample1.jpg',images[20]) 

# Augment data by flipping images and changing sign of steering
augmented_images, augmented_steering_angles = [], []
for image,steering_angle in zip(images, steering_angles):
    augmented_images.append(image)
    augmented_steering_angles.append(steering_angle)
    augmented_images.append(cv2.flip(image,1))
    augmented_steering_angles.append(steering_angle*-1.0)

# sample flipped image
cv2.imwrite('sample2.jpg',augmented_images[24128]) 

# Convert images and steering_angles to numpy arrays for Keras to accept as input    
X_train = np.array(augmented_images)
y_train = np.array(augmented_steering_angles)

# Build Model
model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape = (160, 320, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Conv2D(24,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(200,activation='relu'))
model.add(Dense(100,activation='relu')) 
#model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(loss='mse',optimizer='adam')

# Train model
model.fit(X_train, y_train, epochs=5, verbose=1, validation_split=0.2, shuffle=True)

# Save model
model.save('model.h5')