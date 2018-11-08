# import libraries
import csv
import cv2

# Read and store lines from driving data log
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skips header line in csv file that contains description
    for line in reader:
        lines.append(line)

# Create empty arrays to hold images     
images = []

# For each line in the driving data log, get camera image (left, right and centre) and steering value
for line in lines:
    # Read in images from center, left and right cameras
    image_centre = cv2.imread('data/'+ '/IMG/' + line[0].split('/')[-1])
   
    # Add images and angles to data set
    images.append(image_centre)

# save sample image
cv2.imwrite('sample1.jpg',images[20]) 

# Augment data by flipping images and changing sign of steering
augmented_images = []
for image in images:
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image,1))

# save sample flipped image
cv2.imwrite('sample2.jpg',augmented_images[41])