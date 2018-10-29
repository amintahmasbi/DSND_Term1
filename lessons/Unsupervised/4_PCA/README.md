## Principal Component Analysis (PCA)

PCA is used to **reduce the dimensionality of your data**.

__Latent Features__: Latent features are features that aren't explicitly in your dataset.

**Principal components** are linear combinations of the original features in a dataset that retain the aim to retain the most information in the original data.

You can think of a **principal component** in the same way that you think about a **latent feature**. [link](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues)

There are two main properties of **principal components**:

1. **They retain the most amount of information in the dataset.**  
2. **The created components are orthogonal to one another.**

__IMPORTANT NOTE__: 

- __Standard scale the data__ only when the variables are different units of measurement If you also standardize the variables to variances = 1, this is often called "PCA based on correlations"
-  __Only center the data__ and leave the variances as they are  "PCA based on covariances". (PCA finds the principal components based on the highest variance, so It can be very different from the former) 

Two major parts of PCA:

`I.` The amount of **variance explained by each component**.  This is called an **eigenvalue**.

`II.` The principal components themselves, each component is a vector of weights. **Principal components** are also known as **eigenvectors**.

---

### PCA vs. LDA

Both Linear Discriminant Analysis (LDA) and PCA are linear transformation methods. PCA yields the directions (principal components) that maximize the variance of the data, whereas LDA also aims to find the directions that maximize the separation (or discrimination) between different classes, which can be useful in pattern classification problem (PCA “ignores” class labels).  **In other words, PCA projects the entire dataset onto a different feature (sub)space, and LDA tries to determine a suitable feature (sub)space in order to distinguish between patterns that belong to different classes.**

#### PCA in sklearn

```python
from sklearn.decomposition import PCA

X = StandardScaler().fit_transform(data)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

# Captured (Explained) Variance Per Principal Component
pca.explained_variance_ratio_
```

