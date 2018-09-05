### Decision Trees

Rely on Entropy
$$
entropy: H = - \sum_{i=1}^n p_i \times log_2(p_i)
$$
Information Gain (IG): synonym to KL-divergence
$$
IG(T,a) = H(T) - H(T|a)
$$
Read Wikipedia for more info

#### Hyper-parameters

1. Maximum Depth
2. Minimum number of samples per leaf/split
3. Maximum number of (allowed) features

#### Decision Trees in sklearn

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_values, y_values)
```

__Hyper-parameters__:

- `max_depth`: The maximum number of levels in the tree.
- `min_samples_leaf`: The minimum number of samples allowed in a leaf.
- `min_samples_split`: The minimum number of samples required to split an internal node.
- `max_features` : The number of features to consider when looking for the best split.

