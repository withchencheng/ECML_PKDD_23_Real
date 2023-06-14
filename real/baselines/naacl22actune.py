
import numpy as np
from collections import Counter
from torch.nn import functional as F
import faiss
from sklearn.cluster import MiniBatchKMeans
import copy             
from real.conferr.utils import compute_entropy

def calc_entropy(x):
    # x is the number of occurrences of each label
    lst = []
    for y in x:
        lst.append(x[y])
    lst = np.array(lst) / np.max(lst) 
    return -np.sum(lst * np.log(lst + 1e-12))


def naacl22actune(args,n_sample, unlabeled_feat, unlabeled_pred):
    # semi supervised self training
    ncentroids = 25
    sample_per_group=10
    beta = 1
    weight = True
    entropy = compute_entropy(unlabeled_pred)
    d = unlabeled_feat.shape[-1]
    if weight: # use weighted K-Means Clustering
        kmeans = MiniBatchKMeans(n_clusters = ncentroids, random_state=0, n_init=3, max_iter=100)# default is k-means++
        kmeans.fit(unlabeled_feat, sample_weight = copy.deepcopy(entropy))
        index = faiss.IndexFlatL2(d)
        index.add(kmeans.cluster_centers_)
        D, I = index.search(unlabeled_feat, 1)
    else:
        kmeans = faiss.Clustering(int(d), ncentroids)
        index = faiss.IndexFlatL2(d)
        kmeans.train(unlabeled_feat, index)
        centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
        index.add(centroid)
        D, I = index.search(unlabeled_feat, 1)
    I = I.flatten()# cluster id of every unlabeled instance
    unlabeled_pseudo = np.argmax(unlabeled_pred, axis = 1)
    scores = []
    indexes = []
    for i in range(ncentroids): # cluster
        idx = (I == i)
        cnt = Counter()
        # calculate the mean entropy of samples
        mean_entropy = np.mean(entropy[idx])
        for z in unlabeled_pseudo[idx]:
            cnt[z] += 1
        # calculate the mean entropy of pseudo labels
        class_entropy = calc_entropy(cnt)
        value = mean_entropy + beta * class_entropy
        scores.append(value)
        sorted_idx = np.argsort(entropy[idx]) 
        idxs = np.arange(len(I))[idx][sorted_idx]                     
        indexes.append(list(idxs)) # indexed[i]   cluster-i ， entropy unlabeled instance ID
    iranks = []
    remains = n_sample
    for i in np.argsort(scores)[::-1]: # loop through clusters by cluster score
        if args.task == "sst-2":
            topK = 10
        else:
            topK = 20
        boundary = -min(remains, sample_per_group, len(indexes[i])//topK)
        iranks += indexes[i][boundary:]
        indexes[i] = indexes[i][:boundary]
        remains -= len( indexes[i][-min(remains, sample_per_group, len(indexes[i])//topK):])
        if remains <= 0:
            break 
    for y in indexes:
        iranks += y
    return iranks, entropy
