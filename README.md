## The feature selection based bisecting K-means. 
Implemented bisecting K-means in Python, with the feature selection. Gradually reduce the feature dimension when the cluster size is smaller.




## Feature Selection:
The feature selection is done by applying PCA to the features and reduce the dimensionality of features gradually. 
The dimension is positive correlated with the clsuter size.

## Pipeline:
The baselien K-Means is from SKLearn.
The bisecting K-means is a top-down clustering model, it starts with all in one cluster. Each time we apply K-Means to the cluster with the largest square distance, with k = 2.

## Evaluation:
The silhouette scores analysis is printed at each time K-Means divide the cluster into two sub clusters.


## Usage:
Simply change the file path in main, and it will read your feature.
