# 0x00. Dimensionality Reduction
 
## 0. PCA
Function def pca(X, var=0.95): that performs PCA on a dataset

## 1. PCA v2
Function def pca(X, ndim): that performs PCA on a dataset

## 2. Initialize t-SNE
Function def P_init(X, perplexity): that initializes all variables required to calculate the P affinities in t-SNE

## 3. Entropy
Function def HP(Di, beta): that calculates the Shannon entropy and P affinities relative to a data point

## 4. P affinities
Function def P_affinities(X, tol=1e-5, perplexity=30.0): that calculates the symmetric P affinities of a data set

## 5. Q affinities
Function def Q_affinities(Y): that calculates the Q affinities

## 6. Gradients
Function def grads(Y, P): that calculates the gradients of Y

## 7. Cost
Function def cost(P, Q): that calculates the cost of the t-SNE transformation

## 8. t-SNE
Function def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500): that performs a t-SNE transformation

