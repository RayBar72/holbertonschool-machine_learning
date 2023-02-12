# 0x02. Tensorflow #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is tensorflow?
- What is a session? graph?
- What are tensors?
- What are variables? constants? placeholders? How do you use them?
- What are operations? How do you use them?
- What are namespaces? How do you use them?
- How to train a neural network in tensorflow
- What is a checkpoint?
- How to save/load a model with tensorflow
- What is the graph collection?
- How to add and get variables from the collection

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. Placeholders | Write the function def create_placeholders(nx, classes): that returns two placeholders, x and y, for the neural network | 0-create_placeholders.py |
| 1. Layers | Write the function def create_layer(prev, n, activation) | 1-create_layer.py |
| 2. Forward Propagation | Write the function def forward_prop(x, layer_sizes=[], activations=[]): that creates the forward propagation graph for the neural network | 2-forward_prop.py |
| 3. Accuracy | Write the function def calculate_accuracy(y, y_pred): that calculates the accuracy of a prediction | 3-calculate_accuracy.py |
| 4. Loss | Write the function def calculate_loss(y, y_pred): that calculates the softmax cross-entropy loss of a prediction | 4-calculate_loss.py |
| 5. Train_Op | Write the function def create_train_op(loss, alpha): that creates the training operation for the network | 5-create_train_op.py |
| 6. Train | Write the function def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"): that builds, trains, and saves a neural network classifier | 6-train.py |
| 7. Evaluate | Write the function def evaluate(X, Y, save_path): that evaluates the output of a neural network | 7-evaluate.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/)


**Project Required by**: HolbertonSchool
