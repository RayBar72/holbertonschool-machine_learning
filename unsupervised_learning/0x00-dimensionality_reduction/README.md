# 0x00. Dimensionality Reduction #

<img src="https://github.com/RayBar72/holbertonschool-machine_learning/blob/master/image.png" width="1000" height="450">

## Learning Objectives ##

- What is eigendecomposition?
- What is singular value decomposition?
- What is the difference between eig and svd?
- What is dimensionality reduction and what are its purposes?
- What is principal components analysis (PCA)?
- What is t-distributed stochastic neighbor embedding (t-SNE)?
- What is a manifold?
- What is the difference between linear and non-linear dimensionality reduction?
- Which techniques are linear/non-linear?

## Content Table ##

| Task | Description | File |
| ----------- | ----------- | ----------- |
| 0. PCA | Write a function def pca(X, var=0.95): that performs PCA on a dataset | 0-pca.py |
| 1. PCA v2 | Write a function def pca(X, ndim): that performs PCA on a dataset | 1-pca.py |
| 2. Initialize t-SNE | Write a function def P_init(X, perplexity): that initializes all variables required to calculate the P affinities in t-SNE | 2-P_init.py |
| 3. Entropy | Write a function def HP(Di, beta): that calculates the Shannon entropy and P affinities relative to a data point | 3-entropy.py |
| 4. P affinities | Write a function def P_affinities(X, tol=1e-5, perplexity=30.0): that calculates the symmetric P affinities of a data set | 4-P_affinities.py |
| 5. Q affinities | Write a function def Q_affinities(Y): that calculates the Q affinities | 5-Q_affinities.py |
| 6. Gradients | Write a function def grads(Y, P): that calculates the gradients of Y | 6-grads.py |
| 7. Cost | Write a function def cost(P, Q): that calculates the cost of the t-SNE transformation | 7-cost.py |
| 8. t-SNE | Write a function def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500): that performs a t-SNE transformation | 8-tsne.py |

## Authors: ##

**Solution by:** Raymundo Barrera Flores. [rbarreraf72@gmail.com](rbarreraf72@gmail.com)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>](https://www.linkedin.com/in/raymundo-barrera-flores-a13022222/)


**Project Required by**: HolbertonSchool
