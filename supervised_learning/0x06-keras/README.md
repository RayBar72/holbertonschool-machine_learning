# 0x06. Keras #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is Keras?
- What is a model?
- How to instantiate a model (2 ways)
- How to build a layer
- How to add regularization to a layer
- How to add dropout to a layer
- How to add batch normalization
- How to compile a model
- How to optimize a model
- How to fit a model
- How to use validation data
- How to perform early stopping
- How to measure accuracy
- How to evaluate a model
- How to make a prediction with a model
- How to access the weights/outputs of a model
- What is HDF5?
- How to save and load a model’s weights, a model’s configuration, and the entire model

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. Sequential | Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library | 0-sequential.py |
| 1. Input | Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library | 1-input.py |
| 2. Optimize | Write a function def optimize_model(network, alpha, beta1, beta2): that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics | 2-optimize.py |
| 3. One Hot | Write a function def one_hot(labels, classes=None): that converts a label vector into a one-hot matrix | 3-one_hot.py |
| 4. Train | Write a function def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False): that trains a model using mini-batch gradient descent | 4-train.py |
| 5. Validate | Based on 4-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False): to also analyze validaiton data | 5-train.py |
| 6. Early Stopping | Based on 5-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False): to also train the model using early stopping | 6-train.py |
| 7. Learning Rate Decay | Based on 6-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False): to also train the model with learning rate decay | 7-train.py |
| 8. Save Only the Best | Based on 7-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False): to also save the best iteration of the model | 8-train.py |
| 9. Save and Load Model | Write the following functions | 9-model.py |
| 10. Save and Load Weights | Write the following functions | 10-weights.py |
| 11. Save and Load Configuration | Write the following functions | 11-config.py |
| 12. Test | Write a function def test_model(network, data, labels, verbose=True): that tests a neural network | 12-test.py |
| 13. Predict | Write a function def predict(network, data, verbose=False): that makes a prediction using a neural network | 13-predict.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/)


**Project Required by**: HolbertonSchool
