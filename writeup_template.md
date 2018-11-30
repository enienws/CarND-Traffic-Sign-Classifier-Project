# **Traffic Sign Recognition** 
[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"


[image13]: ./new_data/1.png "Traffic Sign 1"
[image14]: ./new_data/2.png "Traffic Sign 2"
[image15]: ./new_data/3.png "Traffic Sign 3"
[image16]: ./new_data/4.png "Traffic Sign 4"
[image17]: ./new_data/5.png "Traffic Sign 5"
[image18]: ./new_data/6.png "Traffic Sign 6"
[image19]: ./new_data/7.png "Traffic Sign 7"
[image20]: ./new_data/8.png "Traffic Sign 8"
[image21]: ./new_data/new_world_set.png "Traffic Sign 9"
[image23]: ./new_data/new_test_world_result.png "Traffic Sign 11"
[image24]: ./images/data_visualization.png "Visualization"
[image25]: ./images/data_visualization_after.png "Visualization After"

[image9]: ./images/rotation.png "Augment Example 1"
[image10]: ./images/rotation2.png "Augment Example 2"
[image11]: ./images/rotation3.png "Augment Example 3"
[image12]: ./images/gray_scale.png "Gray Scale Example"


### Writeup / README

HTML output of ipython notebook can be found in following link:  [link](https://github.com/enienws/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
Additionally [project code](https://github.com/enienws/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) can be found on this link.
I forked the original CarND-Traffic-Sign-Classifier-Project and make the modifications in my forked repository. Every work including  trained models, images and this markdown file can be found in this repository.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library and basic Python function in order to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I plotted total samples vs. classes graph. As one can easily note, there are huge differences between total samples belonging to different classes. This may hurt the performance of trained model, since model tends to learn better some classes. I will work on balancing data between classes on pre processing section. Detailed work can be found in related section. 

![alt text][image24]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I have implemented sevral ways for preprocessing purposes. Below one can find the methods:
**1. SimplePreprocess**: Simple preprocess is just moves data to zero mean and transforms it in a way so data has unit variance in its all axes, such that red, green, blue channels. Instead of calculating actual mean and variance for data, I just used the simple formula defined by: 
(value - 128) / 128 
In this formula it is assuming that, data possibly have 128 mean and it has a normal distribution such that its variance is 128 for all color channels.
**2. SimplePreprocessGrayscale**: This method applies simple preprocessing technique defined in **SimplePreprocess** method but converting each image to grayscale before applying the data normalization. For grayscale normalization the following formula is used:
Gray = 0.299 x Red + 0.587 x Green  + 0.114 x Blue
This formula is implemented as a matrix vector dot product by using numpy. 
Below an example can be found for color to grayscale conversion result:
![alt text][image12]
**3. AugmentData**: This method augments data, and after augmentation other defined preprocess methods are applied on data. Augmentation is applied in such a way that each class has nearly equal number of images. For augmentation, original images are rotated randomly. Rotation angle is minimum 5 degrees and maximum 15 degrees. 
![alt text][image9]
![alt text][image10]
![alt text][image11]

After augmentation of data new samples for classes is given in the following list:
1260 1980 2010 1260 1770 1650 1800 1290 1260 1320 1800 1170 1890 1920  690 1620 1800  990 1080 1260 1500 1350 1650 1350 1680 1350 1620 1470 1440 1680 1950  690 1470 1797 1800 1080 1650 1260 1860 1350 1500 1470 1470

Moreover data visualization is found on previous section.

 Before augmentation samples for separate classes are given in following list:
180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920  690 540  360  990 1080  180  300  270  330  450  240 1350  540  210  480  240 390  690  210  599  360 1080  330  180 1860  270  300  210  210
The visulization graph after data augmentation is found below image. It seems better however some more work needed here. 

![alt text][image25]

It can be seen that, before augmentation sample count difference between classes is much more striking. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 8x8x24 	|
| Max pooling	      	| 2x2 stride,  outputs 8x8x24 				|
| RELU					|												|
| Fully connected 1		| Input: 1536, Output 120        									|
| Fully connected 2		| Input: 120, Output 84        									|
| Fully connected 3		| Input: 84, Output 43        									|
| Softmax				|         									|
 
Network consists of three convolution layers, max pooling is applied after first and third convolution layers, ReLU is used as a non-linearity source. After the convolution layers, three Fully Connected layers are used, 1536 inputs are decreased to 43 which is in turn number of total classes in German Traffic Sign Dataset.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I have used several configurations in order to increase the accuracy. The following parameters are used in tuning operation:
1. Data preprocess
2. Batch Size
3. Epoch Size
4. Learning Rate
5. Loss Function
6. Network Architecture

The following table summarizes all tried configurations:

| Conf # 	| Data Preprocess 	| Batch Size 	| Epoch Size 	| Learning Rate 	| Network Architecture 	| Accuracy 	|
|:------:	|:--------------------------------------:	|:----------:	|:----------:	|:-------------:	|:--------------------:	|:--------:	|
| 1 	| No 	| 128 	| 100 	| 0.001 	| 1 	| N/A 	|
| 2 	| No 	| 128 	| 100 	| 0.0005 	| 1 	| N/A 	|
| 3 	| No 	| 256 	| 100 	| 0.001 	| 1 	| N/A 	|
| 4 	| No 	| 512 	| 100 	| 0.001 	| 1 	| N/A 	|
| 5 	| SimplePreprocess 	| 512 	| 100 	| 0.0005 	| 1 	| 0.905 	|
| 6 	| SimplePreprocess 	| 512 	| 200 	| 0.0005 	| 1 	| 0.911 	|
| 7 	| Grayscale, SimplePreprocess 	| 512 	| 200 	| 0.0005 	| 2 	| 0.902 	|
| 8 	| Grayscale, SimplePreprocess 	| 256 	| 200 	| 0.0001 	| 2 	| 0.867 	|
| 9 	| Grayscale, SimplePreprocess 	| 1024 	| 250 	| 0.0005 	| 2 	| 0.893 	|
| 10 	| Grayscale, SimplePreprocess 	| 1024 	| 250 	| 0.0001 	| 2 	| 0.878 	|
| 11 	| Grayscale, SimplePreprocess 	| 512 	| 200 	| 0.0001 	| 2 	| 0.876 	|
| 12 	| Grayscale, SimplePreprocess 	| 512 	| 200 	| 0.0001 	| 2 	| 0.879 	|
| 13 	| Augmented, Grayscale, SimplePreprocess 	| 512 	| 200 	| 0.0001 	| 2 	| 0.877 	|
| 14 	| Augmented, SimplePreprocess 	| 512 	| 200 	| 0.0001 	| 3 	| 0.895 	|
| 15 	| SimplePreprocess 	| 512 	| 200 	| 0.001 	| 3 	| 0.920 	|
| 16, 18	| SimplePreprocess 	| 512 	| 200 	| 0.001 	| 4 	| 0.950 	|
| 17 	| Augmented, SimplePreprocess 	| 512 	| 200 	| 0.001 	| 4 	| 0.942 	|



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 0.969
* test set accuracy of 0.950

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture used is architecture taken from the lecture, applying LeNet to traffic sign recognition. 
* What were some problems with the initial architecture?
I have tried several different configurations with initial architecture, however It couldn't be able to achieve accuracy rates better than 0.9. Inspecting validation accuracy vs. epoch number I didn't find any overfitting problem. However it seems the architecture underfits since better accuracy levels cannot be obtained even data size is increased. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Since I suspect from underfitting, I decided to change the network architecture. I choose to add another convolutional layer, since it is known that, convolution operators are talented for extracting features that are special for images such as shape, color, edges, etc. In the table above, it can be seen that, test accuracy is increased when switching from configuration # 7 to configuration #15. This means that adding convolutional layers is the correct way in order to achieve more accuracy levels. 
* Which parameters were tuned? How were they adjusted and why?
Batch size, epoch size and learning rate hyperparameters are tuned. Learning rate is tuned because there is a possibility that optimizer could miss better configurations due to the high learning rates. I tuned learning rate and epoch size together. Decreasing learning rate I have increased epoch size to give a chance to optimizer in order to find better configurations. I tried to tune batch size independently from other hyper parameters, however I cannot find a meaningful change between accuracies so that I fixed that hyper parameter after a while. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
In this architectural design, the most important decision is to use of convolutional layers. Convolutinal layers are successful on extracting patterns from images such as, edges, shapes, color information etc. Moreover convolutional neural networks fit best to problems in which input size is static. Since sample images are static in size, ie they have fixed with height and channel size, CNNs fit best for this kind of problems. The same phenomena can be supported as using RNNs for language processing or sound recognition problems. 

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image13] ![alt text][image14] ![alt text][image15] ![alt text][image16]
![alt text][image17] ![alt text][image18]![alt text][image19] ![alt text][image20]

The source is past graduates of this nanodegree:
http://jeremyshannon.com/2017/01/13/udacity-sdcnd-traffic-sign-classifier.html

First of all for all images that can be problem for the classifier that the input images are not from the same distribution of training dataset. It is important to prepare a training set that reflects the real world test dataset. More formally, training and test dataset should be drawn from same distribution. 

For example, if train dataset contains traffic signs that do not reflect the real world data, the classifier won't be able to perform well on real world data. Because classifier generalized a different kind of data. 

As I have mentioned in first paragraph it is important to use train and test data drawn from similar distributions. So let me now discuss how data distribution changes with different properties of images. Changing of properties like brightness, clutter, orientation and scaling changes distribution of a data source. Hence we perform data preprocess in order to normalize data drawn from different sources. For example, brightness can cause a problem if a classifier is trained mostly with darker images but is testing with brighter images. However with a data preprocess phase in which mean subtraction is performing we make distributions of two datasets similar to each other. 

Additionally different image resolutions and sizes does not affect the performance of classifier if high resolution images are being used. Since the input size of the neural network is 32x32, scaling input images to 32x32 does not affect performance. However, one should take care of not to change image aspect ratios (scaling) when scaling the image to 32x32.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image Class			        |     Prediction Class | 
|:---------------------:|:---------------------------------------------:| 
| 3      		| 3   									| 
| 38     			| 38 										|
| 11					| 11											|
| 34	      		| 34					 				|
| 1			| 1      							|
| 18			| 18      							|
| 12			| 12      							|
| 25			| 25      							|

The column on the left classes of input images and the column on the right holds the predicted classes for the given images. A simple function is coded in notebook in order to calculate the accuracy of the test. Since all the predicted classes are correct for these 8 images one cay claim that accuracy is 100%.

I have decided to try another world test set.  The images can be seen below:

![alt text][image21]

The prediction classes are:

![alt text][image23]

Here one evaluates the accuracy as: 
Accuracy = 5 / 9 = 0,55

When two tests are merged we got accuracy:
Accuracy = 13 / 17 = 0.764

which points to a 19.5% performance drop by using test data accuracy 0.950.

Actually a 19.5% performance drop shows me that classifer perform bad when classifier is tested with data drawn from different data source. This may be a sign of unsufficient preprocess (normalization) process or train dataset does not generalize real world data set properly. So I think that the model is not highly confident to deal with much more images. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| 1st Prediction Prob & Class         	| 2nd Prediction Prob & Class | 3rd Prediction Prob & Class | 4th Prediction Prob & Class | 5th Prediction Prob & Class |
|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| 9.99423265e-01 & 3       	| 4.23493650e-04 & 5 | 4.85798591e-05 & 2 | 4.75823035e-05 & 31 |   4.57842034e-05 & 10 |
| 1.00000000e+00 & 38    	 | 1.17224002e-15 & 25 | 3.29172678e-18 & 13 | 2.88525850e-18 & 36  | 4.98181777e-19 & 1 |
| 1.00000000e+00 & 11	| 5.00808284e-08 & 30 |   2.08032502e-09 & 28 | 6.90515492e-11 & 25 |   4.28188596e-11 & 27 |
| 1.00000000e+00 & 34	| 2.94839869e-12 & 38 | 1.37637308e-12 & 35 | 4.08676006e-16 & 13 |  1.05464370e-16 & 36 |
| 9.99904990e-01 & 1	| 9.14954144e-05 & 2 |   1.50227368e-06 & 7 | 7.20539731e-07 & 5 | 7.14688269e-07 & 0 |
| 1.00000000e+00 & 18	|  4.19746948e-09 & 26 |  1.28743847e-12 & 15 | 5.54263558e-13 & 24 | 3.26915415e-13 & 29 |
| 1.00000000e+00 & 12	| 1.00818822e-13 & 42 |  7.50035478e-14 & 40 | 4.49237205e-14 & 32 |  2.02623071e-14 & 26 |
| 9.78318632e-01 & 25	| 1.64712500e-02 & 1 | 5.12215029e-03 & 5 | 7.09153755e-05 & 4 | 1.17992713e-05 & 14 |


Above in the table top 5 softmax outputs for each image. Softmax outputs class predictions for each class and sum of each probability for class of an image equals to one. 

The second test dataset is shown in image below:

![alt text][image23]


| 1st Prediction Prob & Class | 2nd prediction Prob & Class | 3rd Prediction prob & Class | 4th prediction prob & Class | 5th prediction Prob & Class |
|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| 9.99303222e-01 & 25   | 5.37821674e-04 & 22 | 1.53720946e-04 & 36 | 3.15596344e-06 & 13 | 1.87519652e-06 & 28 |
| 1.00000000e+00 & 17  	|1.10890304e-10 & 20 | 5.90167498e-11 & 14 | 1.73156003e-11 & 18 | 9.85106528e-13 & 23 |
| 7.40100503e-01 & 40	|2.51446068e-01 & 24| 6.16378477e-03 & 4 | 2.19199201e-03 & 1 | 2.37040567e-05 & 37 |
| 9.99999881e-01 & 14	| 1.55160549e-07 & 1 | 9.65049196e-09 & 17 | 1.90187532e-09 & 13 | 1.55462920e-10 & 25 |
| 7.67329216e-01 & 8	| 1.33771434e-01 & 0 | 3.47548313e-02 & 1 | 3.40694375e-02 & 2 | 2.80069839e-02 &9 |
| 9.02192652e-01 & 18	| 5.05776592e-02 & 29 | 4.65159602e-02 & 25 | 3.48630885e-04 & 31 | 1.67455204e-04 & 24 |
| 9.95065391e-01 & 25	| 3.47388093e-03 & 14| 1.10137637e-03 & 1 | 1.26148661e-04 & 5 | 9.26450739e-05 & 31 |
| 9.58731055e-01 & 25	| 1.94614716e-02 & 28 | 1.11226775e-02 & 22 | 5.59002301e-03 & 10 | 4.68781311e-03 & 3 |
| 9.99935031e-01 & 14	|3.53862706e-05 & 1| 2.80956792e-05 & 5 | 9.60402531e-07 & 25 | 2.94951946e-07 & 7 |

This test shows that, distribution of training set is so important. One should generate training set so that it should mimic the real data that will be fed to the model. The selected model for this example is not trained with augmented data so it is actually a mistake for me to choose a model that exceeds 0.93 test accuracy threshold however does not trained with augmented data. I think that, augmented data will much more mimic the real world test dataset. 

Secondly, maybe the cropping of the second world dataset is problematic which causes a lower accuracy rate. Maybe cropping the dataset using the bounding boxes and then training a model will be much more accurate. Moreover I think that this model maybe perfom better on real test dataset. 