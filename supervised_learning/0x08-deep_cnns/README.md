# 0x08. Deep Convolutional Architectures #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is a skip connection?
- What is a bottleneck layer?
- What is the Inception Network?
- What is ResNet? ResNeXt? DenseNet?
- How to replicate a network architecture by reading a journal article

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. Inception Block | Write a function def inception_block(A_prev, filters): that builds an inception block as described in Going Deeper with Convolutions (2014) | 0-inception_block.py |
| 1. Inception Network | Write a function def inception_network(): that builds the inception network as described in Going Deeper with Convolutions (2014) | 1-inception_network.py |
| 2. Identity Block | Write a function def identity_block(A_prev, filters): that builds an identity block as described in Deep Residual Learning for Image Recognition (2015) | 2-identity_block.py |
| 3. Projection Block | Write a function def projection_block(A_prev, filters, s=2): that builds a projection block as described in Deep Residual Learning for Image Recognition (2015) | 3-projection_block.py |
| 4. ResNet-50 | Write a function def resnet50(): that builds the ResNet-50 architecture as described in Deep Residual Learning for Image Recognition (2015) | 4-resnet50.py |
| 5. Dense Block | Write a function def dense_block(X, nb_filters, growth_rate, layers): that builds a dense block as described in Densely Connected Convolutional Networks | 5-dense_block.py |
| 6. Transition Layer | Write a function def transition_layer(X, nb_filters, compression): that builds a transition layer as described in Densely Connected Convolutional Networks | 6-transition_layer.py |
| 7. DenseNet-121 | Write a function def densenet121(growth_rate=32, compression=1.0): that builds the DenseNet-121 architecture as described in Densely Connected Convolutional Networks | 7-densenet121.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/)


**Project Required by**: HolbertonSchool
