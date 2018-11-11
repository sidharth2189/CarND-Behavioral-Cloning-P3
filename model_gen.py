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

# Read images from the training
def read_image(path):
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # drive.py loads images in rgb
    return image_rgb

# Create a data Generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            # Create empty arrays to hold images and steering values
            images = []
            steering_angles = []
            
            # For each line in the driving data log, read camera image (left, right and centre) and steering value
            for batch_sample in batch_samples:
                image_centre = read_image('data/'+ '/IMG/' + line[0].split('/')[-1])
                image_left = read_image('data/'+ '/IMG/' + line[1].split('/')[-1])
                image_right = read_image('data/'+ '/IMG/' + line[2].split('/')[-1])
                steering_centre = float(batch_sample[3])
                steering_left = steering_centre + steering_correction
                steering_right = steering_centre - steering_correction
                images.extend([image_centre, image_left, image_right])
                steering_angles.extend([steering_centre, steering_left, steering_right])
            
            # Augment training data by flipping images and changing sign of steering
            augmented_images, augmented_steering_angles = [], []
            for image,steering_angle in zip(images, steering_angles):
                augmented_images.append(image)
                augmented_steering_angles.append(steering_angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_steering_angles.append(steering_angle*-1.0) 

            # Convert images and steering_angles to numpy arrays for Keras to accept as input
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# split driving data to train and validate
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Use generator to pull data 
train_generator = generator(train_samples, batch_size=24)
validation_generator = generator(validation_samples, batch_size=24)

# Total number of samples per epoch in training and validation
number_of_cameras = 3 # left, right and centre
number_of_data_augmentations = 2 # images are flipped
epoch_samples_train = number_of_cameras*number_of_data_augmentations*len(train_samples)
epoch_samples_valid = number_of_cameras*number_of_data_augmentations*len(validation_samples)

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
batch_size = 24
model.fit_generator(train_generator, steps_per_epoch=(epoch_samples_train//batch_size),
                   validation_data=validation_generator,validation_steps=(epoch_samples_valid//batch_size), epochs=3, verbose = 1)

# Save model
model.save('model_gen.h5')