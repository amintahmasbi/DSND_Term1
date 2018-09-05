### Naive Bayes

**naive Bayes classifiers** are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

Assumption: Feature are statistically independent

Bayes rule:
$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$
__Advantages that Naive Bayes__:

1. ability to handle an extremely large number of features.
2. it performs well even with the presence of irrelevant features and is relatively unaffected by them. 
3. its relative simplicity. Naive Bayes' works well right out of the box and tuning it's parameters is rarely ever necessary, except usually in cases where the distribution of the data is known. 
4. It rarely ever overfits the data. 
5. its model training and prediction times are very fast for the amount of data it can handle. 

```python
# multinomial Naive Bayes implementation
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(x_train,y_train)
```

