# 0x03. Hyperparameter Tuning

## 0. Initialize Gaussian Process

* Create the class GaussianProcess that represents a noiseless 1D Gaussian process

## 1. Gaussian Process Prediction

Based on 0-gp.py, update the class GaussianProcess:

* Public instance method def predict(self, X_s): that predicts the mean and standard deviation of points in a Gaussian process

## 2. Update Gaussian Process

Based on 1-gp.py, update the class GaussianProcess:

* Public instance method def update(self, X_new, Y_new): that updates a Gaussian Process

## 3. Initialize Bayesian Optimization

Create the class BayesianOptimization that performs Bayesian optimization on a noiseless 1D Gaussian process

* Class constructor def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True)

## 4. Bayesian Optimization - Acquisition

Based on 3-bayes_opt.py, update the class BayesianOptimization:

* Public instance method def acquisition(self): that calculates the next best sample location

## 5. Bayesian Optimization

Based on 4-bayes_opt.py, update the class BayesianOptimization

* Public instance method def optimize(self, iterations=100): that optimizes the black-box function

## 6. Bayesian Optimization with GPyOpt

Write a python script that optimizes a machine learning model of your choice using GPyOpt
