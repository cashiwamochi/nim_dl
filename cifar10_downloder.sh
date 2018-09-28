#!/bin/bash

# Download cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
# extract
tar -zxvf cifar-10-binary.tar.gz
#rename
mv cifar-10-batches-bin cifar10