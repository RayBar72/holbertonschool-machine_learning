# 0x03. Optimization

## 0. Normalization Constants
mandatory
Write the function def normalization_constants(X): that calculates the normalization (standardization) constants of a matrix:

X is the numpy.ndarray of shape (m, nx) to normalize
m is the number of data points
nx is the number of features
Returns: the mean and standard deviation of each feature, respectively
ubuntu@alexa-ml:~/0x03-optimization$ cat 0-main.py 
 
## 1. Normalize
mandatory
Write the function def normalize(X, m, s): that normalizes (standardizes) a matrix:

X is the numpy.ndarray of shape (d, nx) to normalize
d is the number of data points
nx is the number of features
m is a numpy.ndarray of shape (nx,) that contains the mean of all features of X
s is a numpy.ndarray of shape (nx,) that contains the standard deviation of all features of X
Returns: The normalized X matrix

## 2. Shuffle Data
mandatory
Write the function def shuffle_data(X, Y): that shuffles the data points in two matrices the same way:

X is the first numpy.ndarray of shape (m, nx) to shuffle
m is the number of data points
nx is the number of features in X
Y is the second numpy.ndarray of shape (m, ny) to shuffle
m is the same number of data points as in X
ny is the number of features in Y
Returns: the shuffled X and Y matrices
Hint: you should use numpy.random.permutation

## 3. Mini-Batch
mandatory
Write the function def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"): that trains a loaded neural network model using mini-batch gradient descent:

X_train is a numpy.ndarray of shape (m, 784) containing the training data
m is the number of data points
784 is the number of input features
Y_train is a one-hot numpy.ndarray of shape (m, 10) containing the training labels
10 is the number of classes the model should classify
X_valid is a numpy.ndarray of shape (m, 784) containing the validation data
Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing the validation labels
batch_size is the number of data points in a batch
epochs is the number of times the training should pass through the whole dataset
load_path is the path from which to load the model
save_path is the path to where the model should be saved after training
Returns: the path where the model was saved
Your training function should allow for a smaller final batch (a.k.a. use the entire training set)
1) import meta graph and restore session
2) Get the following tensors and ops from the collection restored
x is a placeholder for the input data
y is a placeholder for the labels
accuracy is an op to calculate the accuracy of the model
loss is an op to calculate the cost of the model
train_op is an op to perform one pass of gradient descent on the model
3) loop over epochs:
shuffle data
loop over the batches:
get X_batch and Y_batch from data
train your model
4) Save session
You should use shuffle_data = __import__('2-shuffle_data').shuffle_data
Before the first epoch and after every subsequent epoch, the following should be printed:
After {epoch} epochs: where {epoch} is the current epoch
\tTraining Cost: {train_cost} where {train_cost} is the cost of the model on the entire training set
\tTraining Accuracy: {train_accuracy} where {train_accuracy} is the accuracy of the model on the entire training set
\tValidation Cost: {valid_cost} where {valid_cost} is the cost of the model on the entire validation set
\tValidation Accuracy: {valid_accuracy} where {valid_accuracy} is the accuracy of the model on the entire validation set
After every 100 steps gradient descent within an epoch, the following should be printed:
\tStep {step_number}: where {step_number} is the number of times gradient descent has been run in the current epoch
\t\tCost: {step_cost} where {step_cost} is the cost of the model on the current mini-batch
\t\tAccuracy: {step_accuracy} where {step_accuracy} is the accuracy of the model on the current mini-batch
Advice: the function range can help you to handle this loop inside your dataset by using batch_size as step value

## 4. Moving Average
mandatory
Write the function def moving_average(data, beta): that calculates the weighted moving average of a data set:

data is the list of data to calculate the moving average of
beta is the weight used for the moving average
Your moving average calculation should use bias correction
Returns: a list containing the moving averages of data

## 5. Momentum
mandatory
Write the function def update_variables_momentum(alpha, beta1, var, grad, v): that updates a variable using the gradient descent with momentum optimization algorithm:

alpha is the learning rate
beta1 is the momentum weight
var is a numpy.ndarray containing the variable to be updated
grad is a numpy.ndarray containing the gradient of var
v is the previous first moment of var
Returns: the updated variable and the new moment, respectively


## 6. Momentum Upgraded
mandatory
Write the function def create_momentum_op(loss, alpha, beta1): that creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm:

loss is the loss of the network
alpha is the learning rate
beta1 is the momentum weight
Returns: the momentum optimization operation

## 7. RMSProp
mandatory
Write the function def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s): that updates a variable using the RMSProp optimization algorithm:

alpha is the learning rate
beta2 is the RMSProp weight
epsilon is a small number to avoid division by zero
var is a numpy.ndarray containing the variable to be updated
grad is a numpy.ndarray containing the gradient of var
s is the previous second moment of var
Returns: the updated variable and the new moment, respectively

## 8. RMSProp Upgraded
mandatory
Write the function def create_RMSProp_op(loss, alpha, beta2, epsilon): that creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm:

loss is the loss of the network
alpha is the learning rate
beta2 is the RMSProp weight
epsilon is a small number to avoid division by zero
Returns: the RMSProp optimization operation

## 9. Adam
mandatory
Write the function def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t): that updates a variable in place using the Adam optimization algorithm:

alpha is the learning rate
beta1 is the weight used for the first moment
beta2 is the weight used for the second moment
epsilon is a small number to avoid division by zero
var is a numpy.ndarray containing the variable to be updated
grad is a numpy.ndarray containing the gradient of var
v is the previous first moment of var
s is the previous second moment of var
t is the time step used for bias correction
Returns: the updated variable, the new first moment, and the new second moment, respectively


## 10. Adam Upgraded
mandatory
Write the function def create_Adam_op(loss, alpha, beta1, beta2, epsilon): that creates the training operation for a neural network in tensorflow using the Adam optimization algorithm:

loss is the loss of the network
alpha is the learning rate
beta1 is the weight used for the first moment
beta2 is the weight used for the second moment
epsilon is a small number to avoid division by zero
Returns: the Adam optimization operation

## 11. Learning Rate Decay
mandatory
Write the function def learning_rate_decay(alpha, decay_rate, global_step, decay_step): that updates the learning rate using inverse time decay in numpy:

alpha is the original learning rate
decay_rate is the weight used to determine the rate at which alpha will decay
global_step is the number of passes of gradient descent that have elapsed
decay_step is the number of passes of gradient descent that should occur before alpha is decayed further
the learning rate decay should occur in a stepwise fashion
Returns: the updated value for alpha
ubuntu@alexa-ml:~/0x03-optimization$ cat 11-main.py

## 12. Learning Rate Decay Upgraded
mandatory
Write the function def learning_rate_decay(alpha, decay_rate, global_step, decay_step): that creates a learning rate decay operation in tensorflow using inverse time decay:

alpha is the original learning rate
decay_rate is the weight used to determine the rate at which alpha will decay
global_step is the number of passes of gradient descent that have elapsed
decay_step is the number of passes of gradient descent that should occur before alpha is decayed further
the learning rate decay should occur in a stepwise fashion
Returns: the learning rate decay operation


## 13. Batch Normalization
mandatory
Write the function def batch_norm(Z, gamma, beta, epsilon): that normalizes an unactivated output of a neural network using batch normalization:

Z is a numpy.ndarray of shape (m, n) that should be normalized
m is the number of data points
n is the number of features in Z
gamma is a numpy.ndarray of shape (1, n) containing the scales used for batch normalization
beta is a numpy.ndarray of shape (1, n) containing the offsets used for batch normalization
epsilon is a small number used to avoid division by zero
Returns: the normalized Z matrix

## 14. Batch Normalization Upgraded
mandatory
Write the function def create_batch_norm_layer(prev, n, activation): that creates a batch normalization layer for a neural network in tensorflow:

prev is the activated output of the previous layer
n is the number of nodes in the layer to be created
activation is the activation function that should be used on the output of the layer
you should use the tf.keras.layers.Dense layer as the base layer with kernal initializer tf.keras.initializers.VarianceScaling(mode='fan_avg')
your layer should incorporate two trainable parameters, gamma and beta, initialized as vectors of 1 and 0 respectively
you should use an epsilon of 1e-8
Returns: a tensor of the activated output for the layer


## 15. Put it all together and what do you get?
mandatory
Complete the script 15-model.py to write the function def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'): that builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization:

Data_train is a tuple containing the training inputs and training labels, respectively
Data_valid is a tuple containing the validation inputs and validation labels, respectively
layers is a list containing the number of nodes in each layer of the network
activation is a list containing the activation functions used for each layer of the network
alpha is the learning rate
beta1 is the weight for the first moment of Adam Optimization
beta2 is the weight for the second moment of Adam Optimization
epsilon is a small number used to avoid division by zero
decay_rate is the decay rate for inverse time decay of the learning rate (the corresponding decay step should be 1)
batch_size is the number of data points that should be in a mini-batch
epochs is the number of times the training should pass through the whole dataset
save_path is the path where the model should be saved to
Returns: the path where the model was saved
Your training function should allow for a smaller final batch (a.k.a. use the entire training set)
the learning rate should remain the same within the an epoch (a.k.a. all mini-batches within an epoch should use the same learning rate)
Before each epoch, you should shuffle your training data
Before the first epoch and after every subsequent epoch, the following should be printed:
After {epoch} epochs: where {epoch} is the current epoch
\tTraining Cost: {train_cost} where {train_cost} is the cost of the model on the entire training set
\tTraining Accuracy: {train_accuracy} where {train_accuracy} is the accuracy of the model on the entire training set
\tValidation Cost: {valid_cost} where {valid_cost} is the cost of the model on the entire validation set
\tValidation Accuracy: {valid_accuracy} where {valid_accuracy} is the accuracy of the model on the entire validation set
After every 100 steps of gradient descent within an epoch, the following should be printed:
\tStep {step_number}: where {step_number} is the number of times gradient descent has been run in the current epoch
\t\tCost: {step_cost} where {step_cost} is the cost of the model on the current mini-batch
\t\tAccuracy: {step_accuracy} where {step_accuracy} is the accuracy of the model on the current mini-batch
Note: the input data does not need to be normalized as it has already been scaled to a range of [0, 1]


## 16. If you can't explain it simply, you don't understand it well enough
mandatory
Write a blog post explaining the mechanics, pros, and cons of the following optimization techniques:

Feature Scaling
Batch normalization
Mini-batch gradient descent
Gradient descent with momentum
RMSProp optimization
Adam optimization
Learning rate decay
Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on LinkedIn.

When done, please add all URLs below (blog post, tweet, etc.)

Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.
