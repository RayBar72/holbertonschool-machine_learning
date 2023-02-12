# 0x07. Convolutional Neural Networks #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is a convolutional layer?
- What is a pooling layer?
- Forward propagation over convolutional and pooling layers
- Back propagation over convolutional and pooling layers
- How to build a CNN using Tensorflow and Keras

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. Convolutional Forward Prop | Write a function def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)): that performs forward propagation over a convolutional layer of a neural network | 0-conv_forward.py |
| 1. Pooling Forward Prop | Write a function def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs forward propagation over a pooling layer of a neural network | 1-pool_forward.py |
| 2. Convolutional Back Prop | Write a function def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)): that performs back propagation over a convolutional layer of a neural network | 2-conv_backward.py |
| 3. Pooling Back Prop | Write a function def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'): that performs back propagation over a pooling layer of a neural network | 3-pool_backward.py |
| 4. LeNet-5 (Tensorflow 1) | 
 | 4-lenet5.py |
| 5. LeNet-5 (Keras) | Write a function def lenet5(X): that builds a modified version of the LeNet-5 architecture using keras | 5-lenet5.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/)


**Project Required by**: HolbertonSchool
