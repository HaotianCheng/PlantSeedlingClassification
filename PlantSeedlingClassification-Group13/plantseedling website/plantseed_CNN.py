import tensorflow as tf
from keras.applications import xception
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import os
import mpl_toolkits.axes_grid1
import matplotlib.pyplot as plt

#%matplotlib inline
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 16

# validation set size
valid_set_size_percentage = 10 # default = 10%

pre_train = True
# Initial operation
load_bf_train=True
load_bf_test=False

take_train_samples= False
take_test_samples= True
num_test_samples= 1

show_plot=False

## read train and test data

# directories
cw_dir = os.getcwd()
data_dir = 'C:\\Galaxy\\Tools\\XM\\htdocs\\plantseedling'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# different species in the data set
species = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
           'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
           'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
num_species = len(species)

# print number of images of each species in the training data



# read all test data
test = []
for file in os.listdir(test_dir):
    test.append(['test/{}'.format(file), file])
test_df = pd.DataFrame(test, columns=['filepath', 'file'])
# print('test_df.shape = ', test_df.shape)

def read_image(filepath, target_size=None):
    img = cv2.imread(os.path.join(data_dir, filepath), cv2.IMREAD_COLOR)
    img = cv2.resize(img.copy(), target_size, interpolation = cv2.INTER_AREA)
    #img = image.load_img(os.path.join(data_dir, filepath),target_size=target_size)
    #img = image.img_to_array(img)
    return img

## detect and segment plants in the image

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output
def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp
def read_segmented_image(filepath, img_size):
    img = cv2.imread(os.path.join(data_dir, filepath), cv2.IMREAD_COLOR)
    img = cv2.resize(img.copy(), img_size, interpolation = cv2.INTER_AREA)

    image_mask = create_mask_for_plant(img)
    image_segmented = segment_plant(img)
    image_sharpen = sharpen_image(image_segmented)
    return img, image_mask, image_segmented, image_sharpen

## read and preprocess all training/validation/test images and labels

def preprocess_image(img):
    img /= 255.
    img -= 0.5
    img *= 2
    return img

target_image_size = 299

# read, preprocess test images  
x_test = np.zeros((len(test_df), target_image_size, target_image_size, 3), dtype='float32')
for i, filepath in tqdm(enumerate(test_df['filepath'])):
    
    # read original image
    #img = read_image(filepath, (target_image_size, target_image_size))
    
    # read segmented image
    _,_,_,img = read_segmented_image(filepath, (299, 299))
    
    # all pixel values are now between -1 and 1
    x_test[i] = preprocess_image(np.expand_dims(img.copy().astype(np.float), axis=0)) 
    
# print('x_train_valid.shape = ', x_train_valid.shape)
# print('x_test.shape = ', x_test.shape)
    
# load xception base model and predict the last layer comprising 2048 neurons per image
base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
x_test_bf = base_model.predict(x_test, batch_size=32, verbose=1)


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def one_hot_to_dense(labels_one_hot):
    num_labels = labels_one_hot.shape[0]
    num_classes = labels_one_hot.shape[1]
    labels_dense = np.where(labels_one_hot == 1)[1]      
    return labels_dense


## neural network with tensorflow

# permutation array for shuffling train data
# perm_array_train = np.arange(len(x_train_bf))
index_in_epoch = 0


# x_size = x_train_bf.shape[1] # number of features
x_size = 2048
y_size = num_species # binary variable
n_n_fc1 = 1024 # number of neurons of first layer
n_n_fc2 = num_species # number of neurons of second layer

# variables for input and output 
x_data = tf.placeholder('float', shape=[None, x_size])
y_data = tf.placeholder('float', shape=[None, y_size])

# 1.layer: fully connected
W_fc1 = tf.Variable(tf.truncated_normal(shape = [x_size, n_n_fc1], stddev = 0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape = [n_n_fc1]))  
h_fc1 = tf.nn.relu(tf.matmul(x_data, W_fc1) + b_fc1)

# add dropout
tf_keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, tf_keep_prob)

# 3.layer: fully connected
W_fc2 = tf.Variable(tf.truncated_normal(shape = [n_n_fc1, n_n_fc2], stddev = 0.1)) 
b_fc2 = tf.Variable(tf.constant(0.1, shape = [n_n_fc2]))  
z_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_data,
                                                                       logits=z_pred));

# optimisation function
tf_learn_rate = tf.placeholder(dtype='float', name="tf_learn_rate")
train_step = tf.train.AdamOptimizer(tf_learn_rate).minimize(cross_entropy)

# evaluation
y_pred = tf.cast(tf.nn.softmax(z_pred), dtype = tf.float32);
y_pred_class = tf.cast(tf.argmax(y_pred,1), tf.int32)
y_data_class = tf.cast(tf.argmax(y_data,1), tf.int32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_class, y_data_class), tf.float32))

# parameters
cv_num = 1 # number of cross validations
n_epoch = 15 # number of epochs
batch_size = 50 
keep_prob = 0.33 # dropout regularization with keeping probability

learn_rate_step = 3 # in terms of epochs

acc_train_DNN = 0
acc_valid_DNN = 0
loss_train_DNN = 0
loss_valid_DNN = 0
y_test_pred_proba_DNN = 0
y_valid_pred_proba = 0

saver = tf.train.Saver()
# use cross validation

with tf.Session() as sess:
    saver.restore(sess, "C:\\Galaxy\\Tools\\XM\\htdocs\\plantseedling\\my_model\\model.ckpt")
    y_test_pred_proba_DNN += y_pred.eval(feed_dict={x_data: x_test_bf, tf_keep_prob: 1.0})


# final test prediction
y_test_pred_proba_DNN /= float(cv_num)
y_test_pred_class_DNN = np.argmax(y_test_pred_proba_DNN, axis = 1)


for n, i in enumerate(y_test_pred_proba_DNN):
    acc = max(i)/sum(i)
    sp = y_test_pred_class_DNN[n]
    if acc > 0.85:
        print('Prediction:', species[sp])
    else:
        print('unknown species')



