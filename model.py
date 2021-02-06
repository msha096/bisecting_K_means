import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
import heapq
import os



def compute_distance(c, cluster, features):
    # cluster: index id to the feature table
    s = 0 
    for i in cluster:
        s += sum(features.loc[i] - c)**2
    return s/len(cluster)


def calculate_s_score(value, label):
    silhouette_df = pd.DataFrame(list(zip(label, value)), columns=['cluster_id','score'])
    res = silhouette_df.groupby('cluster_id').mean()
    print('Silhouette Scores are %f and %f'%(res.loc[0],res.loc[1]))


def k_means(cluster, n, feature_table, selection):
    pca = PCA(n_components = n)
    # features is the global feature with global index
    if selection:
        features = pd.DataFrame(pca.fit_transform(feature_table))
        X = features.loc[cluster]
    else:
        X = feature_table.loc[cluster]
    
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


def linear_n(n, feature_table):
    # n is positive-correlated with cluster size
    cluster_ratio = n/len(feature_table)
    print('cluster_raio',cluster_ratio)
    n = max(3,int(1000 * cluster_ratio))
    print('PCA to '+ str(n))
    return n

def bisecting_kmeans(k, cluster, priority_queue, feature_table, selection):
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
    queue = []
    p = [i for i in range(len(feature))]
    for i in range(2,max_k):
        print('Processing with k = %d\n'%i)
        res = bisecting_kmeans(i, p, queue, feature, selection)


if __name__ == '__main__':
    current_path = os.path.dirname(__file__)
    tf_idf = pd.read_csv(current_path + '/data/tf_idf_1k.csv')
    tunning_k(max_k = 15, feature = tf_idf, selection = True)