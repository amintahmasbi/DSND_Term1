### Naive Bayes

**naive Bayes classifiers** are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

Assumption: Feature are statistically independent (features are uncorrelated from each other)

Bayes rule:
$$
P(Class|Data) = \frac{P(Data|Class)P(Class)}{P(Data)}
$$
P(Class) is called Prior

P(Class|Data) is called Posterior

P(Data|Class) is called likelihood (Gaussian in Gaussian naive Bayes)

P(Data) is called marginal probability (only a normalizing value, so not actually need to be calculated for classifier)

__Advantages that Naive Bayes__:

1. ability to handle an extremely large number of features.
2. it performs well even with the presence of irrelevant features and is relatively unaffected by them. 
3. its relative simplicity. Naive Bayes' works well right out of the box and tuning it's parameters is rarely ever necessary, except usually in cases where the distribution of the data is known. 
4. It rarely ever overfits the data. 
5. its model training and prediction times are very fast for the amount of data it can handle. 

#### Multinomial Naive Bayes

Assumption:  the features are multinomially distributed. In practice, this means that this classifier is commonly used when we have discrete data.

```python
# multinomial Naive Bayes implementation
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(x_train,y_train)
```

#### Bernoulli Naive Bayes Classifier

The Bernoulli naive Bayes classifier assumes that all our features are binary such that they take only two values (e.g. a nominal categorical feature that has been one-hot encoded).

```python
from sklearn.naive_bayes import BernoulliNB
```

#### Gaussian Naive Bayes

Assumption:  the value of the features are normally (gaussian) distributed (Gaussian naive Bayes)

```python
from sklearn.naive_bayes import GaussianNB

# Create Gaussian Naive Bayes object with prior probabilities of each class
clf = GaussianNB(priors=[0.25, 0.25, 0.5])
```

Note: the raw predicted probabilities from Gaussian naive Bayes (outputted using `predict_proba`) are not calibrated. That is, they should not be believed. If we want to create useful predicted probabilities we will need to calibrate them using an isotonic regression or a related method.

##### Calibrate Predicted Probabilities

 In scikit-learn we can use the `CalibratedClassifierCV` class to create well calibrated predicted probabilities using k-fold cross-validation. In `CalibratedClassifierCV` the training sets are used to train the model and the test sets is used to calibrate the predicted probabilities. The returned predicted probabilities are the average of the k-folds.

```python
from sklearn.calibration import CalibratedClassifierCV

# Create Gaussian Naive Bayes object
clf = GaussianNB()
# Create calibrated cross-validation with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
# Calibrate probabilities
clf_sigmoid.fit(X, y)
# View calibrated probabilities
clf_sigmoid.predict_proba(new_observation)
```

