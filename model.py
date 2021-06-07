import pandas as pd 
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score

import matplotlib.pyplot as plt 
import seaborn as sns
import heapq
import os

def compute_distance(c, cluster, features):
    '''
    compute the square distance between each sample and the centroid.
    cluster: index id to the feature table
    features: the global features with index
    '''
    
    s = 0 
    for i in cluster:
        s += sum(features.loc[i] - c)**2
    return s


def calculate_s_score(value, label):
    '''
    Generate the silhouette score, which does not require ground truth labels.
    '''
    silhouette_df = pd.DataFrame(list(zip(label, value)), columns=['cluster_id','score'])
    res = silhouette_df.groupby('cluster_id').mean()
    print('Silhouette Scores are %f and %f'%(res.loc[0],res.loc[1]))


def k_means(cluster, n, feature_table, selection):
    '''
    Baseline K-Means where k is fixed to 2.
    '''
    pca = PCA(n_components = n)
    # feature_table is the global feature with global index
    if selection:
        # features: pca on global features
        features = pd.DataFrame(pca.fit_transform(feature_table))
        X = features.loc[cluster] # X is the local features
    else:
        pca50 = PCA(n_components = 50)
        features = pd.DataFrame(pca50.fit_transform(feature_table))
        X = features.loc[cluster]
    
    kmeans = KMeans(n_clusters = 2, n_init = 30, max_iter = 500)
    kmeans.fit(X)
    
    cluster_labels = kmeans.labels_
    c = kmeans.cluster_centers_
    # evaluate silhouette score
    values = silhouette_samples(X, cluster_labels)
    calculate_s_score(values, cluster_labels)
    
    cluster_id1,cluster_id2 = [],[]
    for i,l in enumerate(cluster_labels):
        if l == 0:
            cluster_id1.append(cluster[i])
        elif l == 1:
            cluster_id2.append(cluster[i])
        
    # comput the squre distance
    
    d1 = compute_distance(c[0],cluster_id1, features)
    d2 = compute_distance(c[1],cluster_id2, features)
   
    return (-d1,cluster_id1), (-d2,cluster_id2)



def linear_n(cluster_size, total_size):
    ''' 
    The algorithm to reduce the feature dimension.
    return: n is positive-correlated with cluster ratio
    '''
    cluster_ratio = cluster_size / len(total_size)
    print('cluster_raio', cluster_ratio)
    n = max(3, int(50 * cluster_ratio))
    print('PCA to '+ str(n))
    return n

def bisecting_kmeans(k, cluster, priority_queue, feature_table, selection):
    '''
    bisecting k-means model
    k: the target total number of clusters at the end
    cluster: a list of sample IDs/Indexes
    priority_queue: a heapq to find the max square distance
    feature_table: the global features
    selection: boolean, weather to use the feature selection or not

    '''
    while k > 1:
        if not priority_queue:
            print('Starting:\n')
            result1, result2 = k_means(cluster, linear_n(len(cluster), feature_table), feature_table, selection)
        else:
            d, cluster = heapq.heappop(priority_queue)
            print('Currently remaining clusters is %d, max distance is %d\n'%(k,abs(d)))
            result1, result2 = k_means(cluster, linear_n(len(cluster), feature_table), feature_table, selection)
        k -= 1
        heapq.heappush(priority_queue, result1)
        heapq.heappush(priority_queue, result2)
    final_clsuter = []
    for d, cluster_ids in priority_queue:
        final_clsuter.append(cluster_ids)
    return final_clsuter


def tunning_k(max_k, feature, selection):
    '''
    The takeaway method to find the optimal k.
    Looping from 2 to max_k.
    '''
    queue = []
    p = [i for i in range(len(feature))]
    for i in range(2, max_k):
        print('\n')
        print('Processing with k = %d\n'%i)
        res = bisecting_kmeans(i, p, queue, feature, selection)


if __name__ == '__main__':
    current_path = os.path.dirname(__file__)
    tf_idf = pd.read_csv(current_path + '/data/tf_idf_1k.csv')
    # tf_idf.drop('Unnamed: 0', axis = 1, inplace = True)
    # print(tf_idf.head(5))
    tunning_k(max_k = 12, feature = tf_idf, selection = True)
