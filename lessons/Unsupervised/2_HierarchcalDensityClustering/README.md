## Hierarchical Clustering

#### single-link clustering: 

1. assume each point is a cluster
2. find the nearest two cluster and merge/join them
   1. cluster distance definition (smallest distance between two clusters) in _single-link_ is the distance between two closest points in those two clusters 
3. Connect those clusters in the dendrogram (in scipy) (hierarchical-tree-plot)

_Dendrogram_ provides an additional ability to visualize

#### Complete-link clustering:

Similar to single-link, with a different definition for distance between two clusters. It is defined as the farthest two point in the clusters.

### Average-link clustering:

Same as two above while the distance between two clusters is defined as the average of distance between all the points in two clusters.

### Ward's Method:

New definition of distance between two clusters.

1. find the central point between two clusters (average over each feature of all points in two clusters).
2. Add up the squared distance of all points to this center point
3. Reduce the variance inside each cluster from the sum
   1. find the center point of each cluster
   2. Sum up the distance of each point inside that cluster to its center point

Distance between cluster A (p1 and p2) and B (p3 and p4):
$$
\Delta (A, B) = C_{p1}^2 + C_{p2}^2 + C_{p3}^2 + C_{p4}^2 - A_{p1}^2 - A_{p2}^2 - B_{p3}^2 - B_{p4}^2
$$
_Disadvantages of Hierarchical clustering_:

- sensitive to noise and outliers
- computationally intensive! $O(N^2)$

#### Hierarchical (Agglomerative) clustering in sklearn

`adjusted_rand_score`  is an *external cluster validation index* which results in a score between -1 and 1 (1 means two clusterings are identical)

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# Hierarchical clustering
# Ward is the default linkage algorithm
ward = AgglomerativeClustering(n_clusters=3)
ward_pred = ward.fit_predict(data)

ward_ar_score = adjusted_rand_score(target, ward_pred)

# Hierarchical clustering using complete linkage
complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
# Fit & predict
complete_pred = complete.fit_predict(data)

# Hierarchical clustering using average linkage
avg = AgglomerativeClustering(n_clusters=3, linkage='average')
# Fit & predict
avg_pred = avg.fit_predict(data)
```

#### Dendrogram visualization with scipy

```python
# Import scipy's linkage function to conduct the clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

# Specify the linkage type. Scipy accepts 'ward', 'complete', 'average', as well as other values
linkage_type = 'ward'

linkage_matrix = linkage(data, linkage_type)

plt.figure(figsize=(12,18))
# plot using 'dendrogram()'
dendrogram(linkage_matrix)

plt.show()
```

#### Visualization with clustermap (Seaborn)

```python
import seaborn as sns

# Specify the linkage type. Scipy accepts 'ward', 'complete', 'average', as well as other values
linkage_type = 'ward'

sns.clustermap(data, figsize=(12,18), method=linkage_type, cmap='viridis')

plt.show()
```

---

### DBSCAN (Density-based Spatial Clustering of Applications with Noise)

Inputs:

- `Epsilon` as the search (coverage) distance around points 
- `MinPts` the minimum number of points required to form a cluster

check all the points randomly, assign them labels and

Returns:

- Noise point (not enough points around it)
- Core point (has more than `MinPts` around it)
  - Two core points in `Epsilon` distance from each other belong to the same cluster 
- Border point (has a core point in its `Epsilon` distance)

_Advantages_:

- No need to specify the number of clusters
- Flexibility in shape and size of clusters
- Robust to Noise and Outliers

_Disadvantages_

- Border points that are reachable from two clusters go to the one finds them first (random)
- Not good at clusters with varying densities (H-DBSCAN to solve it)

#### DBSCAN in sklearn

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN()

labels = dbscan.fit_predict(data)
```

