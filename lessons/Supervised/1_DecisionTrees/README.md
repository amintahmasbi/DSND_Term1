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
# by default uses gini impurity mrasure 
# model = DecisionTreeClassifier(criterion='gini')
model = DecisionTreeClassifier()

model.fit(x_values, y_values)

# Predict observation's class    
model.predict(X_test)
# View predicted class probabilities for the three classes
model.predict_proba(X_test)
```

__Hyper-parameters__:

- `max_depth`: The maximum number of levels in the tree.
- `min_samples_leaf`: The minimum number of samples allowed in a leaf.
- `min_samples_split`: The minimum number of samples required to split an internal node.
- `max_features` : The number of features to consider when looking for the best split.

#### Decision Tree Regression

Decision tree regression works similar to decision tree classification, however instead of reducing Gini impurity or entropy, potential splits are measured on how much they reduce the mean squared error (MSE).

#### Visualizing Decision Tree

```python
from IPython.display import Image  
from sklearn import tree
import pydotplus

# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())
# Create PDF
graph.write_pdf("iris.pdf")
```

