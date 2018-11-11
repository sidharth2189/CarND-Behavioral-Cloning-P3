# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Original.jpg "Original Model Training"
[image2]: ./images/model.jpg "Model training"
[image3]: ./images/architecture.jpg "Model Architecture"
[image4]: ./images/sample.jpg "Sample Training Image"
[image5]: ./images/sample_flipped.jpg "Sample Flipped Image"
[image6]: ./images/model_gen.jpg "Model with Generator"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The pipeline does not use a generator since not using a generator using the very same model, with same training data yielded succesful result. This is elaborated in "model architecture and training strategy" section.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5X5 filter sizes and depths of 24, 36, 48, 64 and 64 (model.py lines 66-70) 

The model includes RELU layers to introduce nonlinearity (code line 66-70 and 73-76), and the data is normalized in the model using a Keras lambda layer (code line 65). 

Further, not all of the pixels in the input image, contain useful information. To weed out information other than the road, the images are cropped using the Keras Cropping2D layer. This way the model can train faster.

#### 2. Attempts to reduce overfitting in the model

The [original model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) is modified by expanding one of the fully connected layer and then a dropout layers is used between the convolutional layers and the fully connected layers, in order to reduce overfitting (model.py line 72). However the original model, which has no dropout layer, is effective as well.

The model was trained and validated on the sample [data set](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). The model was tested by running it through the simulator on track one. The vehicle was able to succesfully complete a lap in autonomous mode without leaving the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 80).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the training data set provided that drives the vehicle on the centre of the lane.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find model, so that it overfits the data set and then fine-tune the model.

My first step was to use a convolution neural network model similar to the one described in the [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). This model solves end to end learning for self driving cars by detecting road features with human steering angle as training signal. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a mean squared error on the training, close to the one on the validation set. This implied that the model was a good fit already, as shown below. 

![alt text][image1]

However, for purpose of experimenting, a wider network at one of the fully connected layer was chosen to see if the losses could be further minimized. The training went as shown below.

![alt text][image2]

To combat the overfitting, I modified the model by introducting a dropout layer between the set of convolutional and the set of fully connected layers. 

The final step was to run the simulator to see how well the car was driving around track one.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Note that the final model does not use a generator. While training the exact same [model](https://github.com/sidharth2189/CarND-Behavioral-Cloning-P3/blob/master/model_gen.py), that uses a generator went as below, the vehicle could not succesfully complete track one and drove off the track as depicted [here](https://github.com/sidharth2189/CarND-Behavioral-Cloning-P3/blob/master/video_gen.mp4).

![alt text][image6]

#### 2. Final Model Architecture

The final model architecture (model.py lines 63-77) consisted of a convolution neural network with the following layers and layer sizes.

```sh
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
```
![alt text][image3]

#### 3. Creation of the Training Set & Training Process

I used the udacity training data to input mages to my network. An example image is shown below

![alt text][image4]

To augment the data sat, I flipped images and angles thinking that this would generalize the model. For example, here is an image that has then been flipped:

![alt text][image5]

I also used the left and right camera images in addition along with a corrected steering angle value. (steering correction = 0.2)
This can help the vehicle recover if off-centred. Further training data of vehicle weaving into the centre from left and right can be added as augmentation.

After the collection process, I had 48216 number of data points. I then preprocessed this data by normalizing. I divided input pixels by 255 and subtracted the values from 0.5 to mean centre to zero. I converted the images to rgb format since the drive.py file reads rgb and cv2.imread() used in the code reads as BGR.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 since the validation losses did not reduce after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
