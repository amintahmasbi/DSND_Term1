## Ensemble Methods

Ensemble methods combine (or ensemble) simple "weak" models in a way that makes the combination of these models better at predicting than the individual models.

Commonly the "weak" learners are decision trees.

By combining algorithms, we can often build models that perform better by meeting in the middle in terms of bias and variance. 

__Bias-Variance Trade off__: [Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)

**Bias**: When a model has high bias, this means that it doesn't do a good job of bending to the data. An example of an algorithm that usually has high bias is linear regression. Even with completely different datasets, we end up with the same line fit to the data. (related to under-fitting)

- Linear models like the one above is low variance, but high bias.

**Variance**: When a model has high variance, this means that it changes drastically to meet the needs of every point in our dataset.  An example of an algorithm that tends to have a high variance and low bias is a decision tree (especially decision trees with no early stopping parameters). A decision tree, as a high variance algorithm, will attempt to split every point into it's own branch if possible. (related to over-fitting)

- decision tree is a trait of high variance, low bias algorithms - they are extremely flexible to fit exactly whatever data they see.

__Ensemble Methods reduce the variance of a black-box estimator__

Randomness to reduce variance:

- Bootstrap the data: sampling with replacement
- subset the features

#### Random Forest

using bootstrap and subset of features with decision trees makes a random forest

```python
from sklearn.ensemble import RandomForestClassifier
# Create random forest classifer object that uses entropy
clf = RandomForestClassifier(criterion='entropy')
```

##### Train Random Forest While Balancing Classes

When using `RandomForestClassifier` a useful setting is `class_weight=balanced` wherein classes are automatically weighted inversely proportional to how frequently they appear in the data.

```python
clf = RandomForestClassifier(class_weight="balanced")
```

##### Select Important Features In Random Forest

Select Features With Importance Greater Than Threshold. The higher the number, the more important the feature (all importance scores sum to one).

```python
from sklearn.feature_selection import SelectFromModel
# Create random forest classifier
clf = RandomForestClassifier(n_jobs=-1) # parallel training
# Create object that selects features with importance greater than or equal to a threshold
selector = SelectFromModel(clf, threshold=0.3)
# Feature new feature matrix using selector
X_important = selector.fit_transform(X, y)
```

##### Random Forest Regression

```python
from sklearn.ensemble import RandomForestRegressor
```

---

#### Bagging algorithm

 A weak learner on each subset of data (bootstrap). Combine by max voting

#### AdaBoost 

1. Init all samples with weight 1

2. A weak learner to minimize error

3. Increase the weight of incorrect point to balance the correct-incorrect

4. Repeat step 2 and 3 and store learners (until get enough weak learners or meet some other criteria)

5. Combine them by adding the positive and negative values of each learner for each region. Weight based on accuracy:
   $$
   weight_i = ln (\frac{accuracy_i}{1-accuracy_i})
   $$

6. If the sum is positive $y=1$ region, otherwise $y=-1$ .

```python
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(x_train, y_train)
model.predict(x_test)
```

**Hyper-parameters:**

- `base_estimator`: The model (learning algorithm) to use for the weak learners. This will almost always not needed to be changed  because by far the most common learner to use with AdaBoost is a  decision tree – this parameter’s default argument. (**Warning:** Don't forget to import the model that you decide to use for the weak learner).
- `n_estimators`: The maximum number of weak learners  (iteratively) trained.
- `learning_rate` is the contribution of each model to the weights and defaults to `1`.  Reducing the learning rate will mean the weights will be increased or  decreased to a small degree, forcing the model train slower (but sometimes resulting in better performance scores).
- `loss` is exclusive to `AdaBoostRegressor` and sets the loss function to use when updating weights. This defaults to a linear loss function however can be changed to `square` or `exponential`.

#### Gradient Boosting and XGBoost (eXtreme GBoost)

(Not covered in the course)

Iteratively make weak learner (generally a decision tree) stronger:

1. Initialize the model (weak learner) with a constant value. $F_0(x)$ (mean of target value) 

2. For M steps repeat:

   1. Compute *pseudo* residuals (error in prediction with respect to target value)

   2. Fit base learner, $h_m(x)$ to pseudo residuals

   3. Compute step magnitude multiplier $\gamma_m$ . (In the case of tree models, compute a different $\gamma_m$ for every leaf.)

   4. Update model:
      $$
      F_m(x) = F_{m-1}(x) + \gamma_mh_m(x)
      $$

3. http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/

