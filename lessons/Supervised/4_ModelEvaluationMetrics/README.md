## Model Evaluation Metrics

### Classification Metrics:

#### Confusion Matrix

the matrix with all the prediction and label values to calculate all other metrics

Type 1 Error: False Positive

Type 2 Error: False Negative

**Accuracy** measures how often the classifier makes the correct prediction. or ratio of correctly classified to all

_Note_: For classification problems that are skewed (anomaly detection) in their classification distributions accuracy by itself is not a very good metric.

For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score.

**Precision** It is a ratio of true positives to all positives

[True Positives/(True Positives + False Positives)]

**Recall (sensitivity)** ratio of True Positive to all Positives in the dataset (how many points were covered)

[True Positives/(True Positives + False Negatives)] 

**$F_1$ Score**: is the Harmonic mean $\frac{2xy}{x+y}$ (always smaller than mean $\frac{x+y}{2}$)

**$F_\beta$ score**: tend more toward precision or recall by adjusting $\beta$
$$
F_\beta Score=(1+\beta^2)\times\frac{precision\times recall}{\beta^2\times precision + recall}
$$
The higher the $\beta$ the model is more **high recall** (which cannot afford False negatives)

$F_0$ is precision

$F_1$ is harmonic mean

$F_\infty$ is recall 

#### Receiver Operating Characteristic ([ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic))

The area under the curve (AUC) of _True Positive Rate_ versus _False Positive Rate_ for all different classifiers is ROC curve.

_Note_: Predicting all the points as positive gives the (1,1) point and predicting all as negative gives the (0,0) point

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
accuracy_score(y_test, preds)
precision_score(y_test, preds)
recall_score(y_test, preds)
f1_score(y_test, preds)
fbeta_score(y_test, preds, beta=1)

from sklearn.metrics import roc_curve, auc, roc_auc_score
```

---

### Regression Metrics:

MAE: Mean absolute Error $|y-\hat y|$ (not differentiable)

- When the values to predict follow a skewed distribution.
- Particularly helpful in the cases with outliers as they will not influence models attempting to optimize on this metric.
- The optimal value for this technique is the median value. 

MSE: Mean Squared Error $\frac{1}{2} (y-\hat y)^2$

R2 Score (also $R^2$ or $r^2$): [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) 

the 'amount of variability' captured by a model
$$
R^2 = 1- \frac{\propto MSE_{regression model}}{\propto Var(y)} = 1- \frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}
$$

- When you optimize for the R2 value or the mean squared error, the optimal value is the mean.

Pearson correlation coefficient (Pearson's $r​$): (NOT COVERED IN COURSE) _only for linear models_ 

a measure of the linear [correlation](https://en.wikipedia.org/wiki/Correlation) between two variables $X$ and $Y$
$$
\rho = \frac{cov(X,Y)}{\sigma_X \sigma_Y}
$$
Note that the correlation coefficient reflects the non-linearity and direction of a linear relationship, __but not the slope of that relationship__, nor many aspects of nonlinear relationships.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
r2_score(y_test, preds)
```

