# PlantSeedlingClassification-Group13
Database: https://www.kaggle.com/c/plant-seedlings-classification/data
* In Sprint 2, we load and preprocess the Train Data and Test Data, we detect and segment the plant in the images and get the segmented images.
* In Sprint3, We extract the bottled-neck feartures using Xception Models from keras to improve the accuracy of the model.
* We build two models (logistic regression, a self-written CNN fully connected layers) to train the data we processed and we compare the results. The total accuracy is around 90% but the accuracy of BG and LSB is around 70%.
* In Sprint 4, we build a website for users to upload the images and return the species of the plant seedlings if available.
* In the final sprint, we make some improvement to our previous model that it can train BG and LSB independently. As a result, we increase the accuracy of our model up to 94%.
