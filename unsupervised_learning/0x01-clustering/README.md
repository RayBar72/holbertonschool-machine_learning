# 0x01. Clustering #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is a multimodal distribution?
- What is a cluster?
- What is cluster analysis?
- What is “soft” vs “hard” clustering?
- What is K-means clustering?
- What are mixture models?
- What is a Gaussian Mixture Model (GMM)?
- What is the Expectation-Maximization (EM) algorithm?
- How to implement the EM algorithm for GMMs
- What is cluster variance?
- What is the mountain/elbow method?
- What is the Bayesian Information Criterion?
- How to determine the correct number of clusters
- What is Hierarchical clustering?
- What is Agglomerative clustering?
- What is Ward’s method?
- What is Cophenetic distance?
- What is scikit-learn?
- What is scipy?

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. Initialize K-means | Write a function def initialize(X, k): that initializes cluster centroids for K-means | 0-initialize.py |
| 1. K-means | Write a function def kmeans(X, k, iterations=1000): that performs K-means on a dataset | 1-kmeans.py |
| 2. Variance | Write a function def variance(X, C): that calculates the total intra-cluster variance for a data set | 2-variance.py |
| 3. Optimize k | Write a functiondef optimum_k(X, kmin=1, kmax=None, iterations=1000): that tests for the optimum number of clusters by variance | 3-optimum.py |
| 4. Initialize GMM | Write a function def initialize(X, k): that initializes variables for a Gaussian Mixture Model | 4-initialize.py |
| 5. PDF | Write a function def pdf(X, m, S): that calculates the probability density function of a Gaussian distribution | 5-pdf.py |
| 6. Expectation | Write a function def expectation(X, pi, m, S): that calculates the expectation step in the EM algorithm for a GMM | 6-expectation.py |
| 7. Maximization | Write a function def maximization(X, g): that calculates the maximization step in the EM algorithm for a GMM | 7-maximization.py |
| 8. EM | Write a function def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False): that performs the expectation maximization for a GMM | 8-EM.py |
| 9. BIC | Write a function def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False): that finds the best number of clusters for a GMM using the Bayesian Information Criterion | 9-BIC.py |
| 10. Hello, sklearn! | Write a function def kmeans(X, k): that performs K-means on a dataset | 10-kmeans.py |
| 11. GMM | Write a function def gmm(X, k): that calculates a GMM from a dataset | 11-gmm.py |
| 12. Agglomerative | Write a function def agglomerative(X, dist): that performs agglomerative clustering on a dataset | 12-agglomerative.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/)


**Project Required by**: HolbertonSchool
