#!/bin/sh
# Download the training and test data to train the model

curl https://raw.githubusercontent.com/geisten/mnist-data/master/mnist_test_data.tar.gz -O -J -L
curl https://raw.githubusercontent.com/geisten/mnist-data/master/mnist_train_data.tar.gz -O -J -L

tar -xzvf mnist_test_data.tar.gz
tar -xzvf mnist_train_data.tar.gz