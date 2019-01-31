### Linear Regression

**linear regression** is a linear approach to modelling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables)

**Gradient descent** is a method to optimize your linear models.

$$
w_i \leftarrow  w_i - \alpha \frac{\partial}{\partial w_i}Error
$$

MAE: Mean absolute Error $|y-\hat y|$

MSE: Mean Squared Error $\frac{1}{2} (y-\hat y)^2$

Exact solution can be found instead of gradient descent but it needs to compute the inverse of the n-by-n matrix

_Note_: Linear regression is sensitive to outliers

#### Multiple Linear Regression

A technique for when you are comparing more than two variables.

Note: different from Multivariate Linear Regression, in which you try to model the relationship between more than one target (dependent) variable.

#### polynomial regression

For relationships between variables that aren't linear

Add higher degree features ($x^2$, $x^3$, ...) and use linear regression to find the best polynomial

```python
from sklearn.preprocessing import PolynomialFeatures
poly_feat = PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)
poly_X = poly_feat.fit_transform(X)
```

#### Regularization

A technique to assure that your models will not only fit to the data available, but also extend to new situations

__Lasso__ is linear regression with L1 regularization

__Ridge regression__ is linear regression with L2 norm regularization

The hyperparameter, $\alpha$, lets us control how much we penalize the coefficients, with higher values of $\alpha$ creating simpler modelers. The ideal value of $\alpha$ should be tuned like any other hyperparameter. 

```python
from sklearn.linear_model import Lasso
# Create lasso regression with alpha value
regr = Lasso(alpha=0.5)

# Fit the linear regression
model = regr.fit(X_std, y)
```

```python
from sklearn.linear_model import Ridge
# Create ridge regression with an alpha value
regr = Ridge(alpha=0.5)

# Fit the linear regression
model = regr.fit(X_std, y)
```

##### Selecting The Best Alpha Value In Ridge Regression

scikit-learn includes a `RidgeCV` method that allows us select the ideal value for $\alpha$:

```python
from sklearn.linear_model import RidgeCV
# Create ridge regression with three possible alpha values
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
# Fit the linear regression
model_cv = regr_cv.fit(X_std, y)
# View alpha
model_cv.alpha_
```

##### L1 vs L2 Regularization

|                  L1 Regularization                  |     L2 Regularization     |
| :-------------------------------------------------: | :-----------------------: |
| Computationally Inefficient (unless data is sparse) | Computationally Efficient |
|                   Sparse Outputs                    |    Non-Sparse Outputs     |
|                  Feature Selection                  |   No Feature Selection    |

#### Feature Scaling

1. Standardizing: reduce mean and divide by standard deviation (most common)
2. Normalizing: scale between 0 and 1. reduce min_val and device by the span (max_val-min_val)

_Note_: Use Feature Scaling when:

1. The algorithm uses a distance based metric (e.g., SVM, k-nn)
2. Using regularization

Feature scaling generally speed up the convergence

#### Linear Regression in sklearn

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_values, y_values)
```

