# 0x05. Regularization #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is regularization? What is its purpose?
- What is are L1 and L2 regularization? What is the difference between the two methods?
- What is dropout?
- What is early stopping?
- What is data augmentation?
- How do you implement the above regularization methods in Numpy? Tensorflow?
- What are the pros and cons of the above regularization methods?

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. L2 Regularization Cost | Write a function def l2_reg_cost(cost, lambtha, weights, L, m): that calculates the cost of a neural network with L2 regularization | 0-l2_reg_cost.py |
| 1. Gradient Descent with L2 Regularization | Write a function def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L): that updates the weights and biases of a neural network using gradient descent with L2 regularization | 1-l2_reg_gradient_descent.py |
| 2. L2 Regularization Cost | Write the function def l2_reg_cost(cost): that calculates the cost of a neural network with L2 regularization | 2-l2_reg_cost.py |
| 3. Create a Layer with L2 Regularization | Write a function def l2_reg_create_layer(prev, n, activation, lambtha): that creates a tensorflow layer that includes L2 regularization | 3-l2_reg_create_layer.py |
| 4. Forward Propagation with Dropout | Write a function def dropout_forward_prop(X, weights, L, keep_prob): that conducts forward propagation using Dropout | 4-dropout_forward_prop.py |
| 5. Gradient Descent with Dropout | Write a function def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L): that updates the weights of a neural network with Dropout regularization using gradient descent | 5-dropout_gradient_descent.py |
| 6. Create a Layer with Dropout | Write a function def dropout_create_layer(prev, n, activation, keep_prob): that creates a layer of a neural network using dropout | 6-dropout_create_layer.py |
| 7. Early Stopping | Write the function def early_stopping(cost, opt_cost, threshold, patience, count): that determines if you should stop gradient descent early | 7-early_stopping.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/


**Project Required by**: HolbertonSchool
