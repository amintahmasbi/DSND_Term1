## Clustering

First three steps to identify clusters in a dataset.

1. **Visual Inspection** of the data.
2. **Pre-conceived** ideas of the number of clusters.
3. **The elbow method**, which compares the average distance of each point to the cluster center for different numbers of centers.

### K-means

Partition observations into _k_ clusters, in which each observation belongs to the cluster with the nearest mean.

choosing _k_: Prior/Domain knowledge or Elbow method

_Elbow Method_: create a plot of the number of clusters (on the x-axis) vs. the average distance of the center of the cluster to each point (on the y-axis). This plot is called a **scree plot**. The Elbow is the point where the rate of decrease changes significantly.

Algorithm:

1. randomly assign K centroids
2. Assign points to the nearest centroid
3. Move centroid to the center of the points in that cluster (nearest to it)
4. repeat 2 and 3 until converge

run multiple times to reduce the effect of random selection of centroids.

_Note_: since KMeans has a measure of distance, feature scaling is crucial. 

- Standardizing or __Z-Score Scaling__: mean of 0 and std of 1. (recommended for KMeans)
- Normalizing or __Max-Min Scaling__: linear scale between 0 and 1 (good for images)



#### Kmeans in sklearn

```python
from sklearn.cluster import KMeans
kmeans = KMeans(4)#instantiate your model

# Then fit the model to your data using the fit method
model = kmeans.fit(data)#fit the model to your data

# Finally predict the labels on the same data to show the category that point belongs to
labels = model.predict(data)#predict labels using model on your dataset
```

#### Feature Scaling in sklearn

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaled_data = StandardScaler().fit_transform(data)
scaled_data = MinMaxScaler().fit_transform(data)
```



------

### 