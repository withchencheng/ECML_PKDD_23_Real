from real.conferr.utils import compute_entropy
import numpy as np
import faiss
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
import random



def kmeanspp(ncentroids, feat):
    """
    K-means++
    Args:
      ncentroids (int):
      feat: [n, dim]
    """
    dim = feat.shape[-1]
    kmeans = MiniBatchKMeans(n_clusters = ncentroids, random_state=0, n_init=3, max_iter=100)# default is k-means++
    kmeans.fit(feat)
    index = faiss.IndexFlatL2(dim)
    index.add(kmeans.cluster_centers_)
    D, I = index.search(feat, 1)
    I = I.flatten() # list of cluster assignment for all unlabeled ins
    return  I

def kmeans_cc(ncentroids, feat):
    dim = feat.shape[-1]
    kmeans = faiss.Clustering(int(dim), ncentroids)
    index = faiss.IndexFlatL2(dim)
    kmeans.train(feat, index)
    centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
    index.add(centroid)
    D, I = index.search(feat, 1)
    I = I.flatten() # list of cluster assignment for all unlabeled ins
    return I


def pseudo_err_1_2(args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred):
    """
    ， majority class 
    #sample/cluster: uniform
    Args:
    Return:
    """
    # Use representations of current fine-tuned model *CAL*
    N=unlabeled_pred.shape[0]
    ncentroids = args.ncentroids
    unlabeled_pseudo = np.argmax(unlabeled_pred, axis = -1)
    # r_coh_cluster = args.pe_1_2_r_coh_cluster # good cluster ratio
    entropy = compute_entropy(unlabeled_pred)
    entp_rank = np.argsort(entropy)[::-1] # uncertain 
    sample_idx, save_idx = [], []

    I=kmeanspp(ncentroids, unlabeled_feat)
    topk = n_sample//ncentroids
    for i in range(ncentroids):
        clu_sel = (I == i) # selector for current cluster
        cnt = Counter()
        for z in unlabeled_pseudo[clu_sel]:
            cnt[z] += 1
        # select minority from cnt
        lbl_freq = list(cnt.items())
        if len(lbl_freq)==0: continue # ，faiss
        # print('\n\n\n',lbl_freq )
        lbl_freq.sort(key=lambda x: x[1])
        majlbl = lbl_freq[-1][0] # the majority label
        # value = mean_entropy + beta * class_entropy
        majscore = unlabeled_pred[clu_sel][:, majlbl]
        sorted_idx = np.argsort(majscore)   #clusteridx
        idxs = np.arange(len(I))[clu_sel][sorted_idx]  # unlabeled pool, cluster，entropyidx        
        tmp=[]
        j=0
        while j<len(idxs) and len(tmp)<topk:
            i1=idxs[j]
            tmp.append(i1)
            j+=1
        sample_idx += tmp

    labeled=[False]*N
    for i in sample_idx:
        labeled[i]=True

    i=0
    # ，entropy
    while len(sample_idx)<n_sample:
        j=entp_rank[i]
        if not labeled[j]:
            sample_idx.append(j)
            labeled[j]=True
        i+=1

    for i, lbl in enumerate(labeled):
        if not lbl:
            save_idx.append(i)

    return sample_idx, save_idx, entropy, I


def pseudo_err_1_3(args, n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred):
    """
    ， majority class 
    #sample/cluster: ignore, rank over all unlabeled pool
    Args:
    Return:
    """
    # Use representations of current fine-tuned model *CAL*
    N=unlabeled_pred.shape[0]
    ncentroids = args.ncentroids
    unlabeled_pseudo = np.argmax(unlabeled_pred, axis = -1)
    entropy = compute_entropy(unlabeled_pred)
    sample_idx, save_idx = [], []

    I=kmeanspp(ncentroids, unlabeled_feat)
    dis = np.zeros(N)
    for i in range(ncentroids):
        clu_sel = (I == i) # selector for current cluster
        cnt = Counter()
        for z in unlabeled_pseudo[clu_sel]:
            cnt[z] += 1
        # select minority from cnt
        lbl_freq = list(cnt.items())
        if len(lbl_freq)==0: continue # ，faiss
        # print('\n\n\n',lbl_freq )
        lbl_freq.sort(key=lambda x: x[1])
        majlbl = lbl_freq[-1][0] # the majority label
        majscore = unlabeled_pred[clu_sel][:, majlbl]
        dis[clu_sel]=1-majscore

    ranks = np.argpartition(dis, -n_sample).tolist() 
    sample_idx = ranks[-n_sample:]
    save_idx = ranks[:-n_sample]

    return sample_idx, save_idx, entropy, I


def pseudo_err_1_4(args, n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred):
    """
    ， majority class 
    #sample/cluster: weighted assignment by cluster value, more samples for larger value
    Args:
    Return:
    """
    N=unlabeled_pred.shape[0]
    ncentroids = args.ncentroids
    unlabeled_pseudo = np.argmax(unlabeled_pred, axis = -1)
    entropy = compute_entropy(unlabeled_pred)
    sample_idx, save_idx = [], []

    I=kmeanspp(ncentroids, unlabeled_feat)
    clu_value = [0]*ncentroids # cluster value, more error, more valuable
    clu_majlbl = [-1]*ncentroids # cluster value, more error, more valuable
    dis = np.zeros(N)
    # pass 1: fill clu_value
    for i in range(ncentroids):
        clu_sel = (I == i) # selector for current cluster
        if np.sum(clu_sel)==0: continue # ，faiss
        cnt = Counter()
        for z in unlabeled_pseudo[clu_sel]:
            cnt[z] += 1
        # select minority from cnt
        lbl_freq = list(cnt.items())
        lbl_freq.sort(key=lambda x: x[1])
        clu_pseudo = unlabeled_pseudo[clu_sel]
        majlbl = lbl_freq[-1][0] # the majority label
        clu_majlbl[i] = majlbl
        majscore = unlabeled_pred[clu_sel][:, majlbl]
        dismaj = 1-majscore
        dis[clu_sel]= dismaj
        nonmaj_sel = clu_pseudo!=majlbl # TODO: cluster
        # clu_value[i] = np.mean(dismaj[nonmaj_sel]) # set i， 
        clu_value[i] =  np.sum(dismaj[nonmaj_sel])

    # pass 2: sample proportionanlly to clu_value
    cvsm = np.sum(clu_value)
    if cvsm< 1e-20:
        print('20230309:', cvsm)
        clu_nsample = [0 for i in clu_value]
    else:
        clu_nsample = [int(i/cvsm * n_sample) for i in clu_value]
    nmissing = n_sample - np.sum(clu_nsample)
    highclui=np.argsort(clu_value)[::-1][:nmissing]
    for i in highclui:
        clu_nsample[i]+=1

    for i in range(ncentroids):
        clu_sel = (I == i) # selector for current cluster
        topk = clu_nsample[i] # TODO: topk > clu_size, qqp 
        if topk<=0: continue
        majlbl = clu_majlbl[i]
        majscore = unlabeled_pred[clu_sel][:, majlbl]
        idx = np.argsort(majscore)[:topk]
        tmp = np.arange(len(I))[clu_sel][idx]
        sample_idx += tmp.tolist()

    dis_rank = np.argsort(dis)[::-1] # big 
    i=0
    # ，
    labeled=[False]*N
    for i in sample_idx:
        labeled[i]=True
    while len(sample_idx)<n_sample:
        j=dis_rank[i]
        if not labeled[j]:
            sample_idx.append(j)
            labeled[j]=True
        i+=1

    assert len(sample_idx) == n_sample
    for i, lbl in enumerate(labeled):
        if not lbl:
            save_idx.append(i)

    return sample_idx, save_idx, entropy, I


def pseudo_err_1_4_1(args, n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred):
    """
    CAUTION: sampler  `elif al_method in ['pseudo_err_1.2','pseudo_err_1.3', 'pseudo_err_1.4']: ` 
    ， majority class 
    #sample/cluster: follow 1.4, weighted assignment by cluster value, more samples for larger value
    clusterpseudo err，    combined with random / entropy
     pseudo err probability

    1.4.1 combine with random
    Args:
    Return:
    """
    # Use representations of current fine-tuned model *CAL*
    N=unlabeled_pred.shape[0]
    ncentroids = args.ncentroids
    unlabeled_pseudo = np.argmax(unlabeled_pred, axis = -1)
    entropy = compute_entropy(unlabeled_pred)
    sample_idx, save_idx = [], []

    I=kmeanspp(ncentroids, unlabeled_feat)
    clu_value = [0]*ncentroids # cluster value, more error, more valuable
    clu_majlbl = [-1]*ncentroids # cluster value, more error, more valuable
    dis = np.zeros(N)
    # pass 1: fill clu_value
    for i in range(ncentroids):
        clu_sel = (I == i) # selector for current cluster
        if np.sum(clu_sel)==0: continue # ，faiss
        cnt = Counter()
        for z in unlabeled_pseudo[clu_sel]:
            cnt[z] += 1
        # select minority from cnt
        lbl_freq = list(cnt.items())
        lbl_freq.sort(key=lambda x: x[1])
        clu_pseudo = unlabeled_pseudo[clu_sel]
        majlbl = lbl_freq[-1][0] # the majority label
        clu_majlbl[i] = majlbl
        majscore = unlabeled_pred[clu_sel][:, majlbl]
        dismaj = 1-majscore
        dis[clu_sel]= dismaj
        nonmaj_sel = clu_pseudo!=majlbl
        # clu_value[i] = np.mean(dismaj[nonmaj_sel]) # set i， 
        clu_value[i] =  np.sum(dismaj[nonmaj_sel])

    # pass 2: sample proportionanlly to clu_value
    cvsm = np.sum(clu_value)
    clu_nsample = [int(i/cvsm * n_sample) for i in clu_value]
    nmissing = n_sample - np.sum(clu_nsample)
    highclui=np.argsort(clu_value)[::-1][:nmissing]
    for i in highclui:
        clu_nsample[i]+=1

    for i in range(ncentroids):
        clu_sel = (I == i) # selector for current cluster
        topk = clu_nsample[i] # TODO: topk > clu_size, qqp 
        if topk<=0: continue
        majlbl = clu_majlbl[i]
        clu_pseudo = unlabeled_pseudo[clu_sel]
        majscore = unlabeled_pred[clu_sel][:, majlbl]
        nonmaj_sel = clu_pseudo!=majlbl
        nonmaj_idx = np.arange(len(clu_pseudo))[nonmaj_sel] # 
        npseudoerr = np.sum(nonmaj_sel)
        if npseudoerr > topk: # topk
            # random or entropy pick topk from w
            picki = random.sample(nonmaj_idx.tolist(), topk)
        else:
            picki = np.argsort(majscore)[:topk]
        tmp = np.arange(len(I))[clu_sel][picki]
        sample_idx += tmp.tolist()

    dis_rank = np.argsort(dis)[::-1] # big 
    i=0
    # ，
    labeled=[False]*N
    for i in sample_idx:
        labeled[i]=True
    while len(sample_idx)<n_sample:
        j=dis_rank[i]
        if not labeled[j]:
            sample_idx.append(j)
            labeled[j]=True
        i+=1

    assert len(sample_idx) == n_sample
    for i, lbl in enumerate(labeled):
        if not lbl:
            save_idx.append(i)

    return sample_idx, save_idx, entropy, I


def pseudo_err_1_4_2(args, n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred):
    """
    CAUTION: sampler  `elif al_method in ['pseudo_err_1.2','pseudo_err_1.3', 'pseudo_err_1.4']: ` 
    ， majority class 
    #sample/cluster: follow 1.4, weighted assignment by cluster value, more samples for larger value
    clusterpseudo err，    combined with random / entropy
     pseudo err probability

    1.4.2 combine with entropy
    Args:
    Return:
    """
    # Use representations of current fine-tuned model *CAL*
    N=unlabeled_pred.shape[0]
    ncentroids = args.ncentroids
    unlabeled_pseudo = np.argmax(unlabeled_pred, axis = -1)
    entropy = compute_entropy(unlabeled_pred)
    sample_idx, save_idx = [], []

    I=kmeanspp(ncentroids, unlabeled_feat)
    clu_value = [0]*ncentroids # cluster value, more error, more valuable
    clu_majlbl = [-1]*ncentroids # cluster value, more error, more valuable
    dis = np.zeros(N)
    # pass 1: fill clu_value
    for i in range(ncentroids):
        clu_sel = (I == i) # selector for current cluster
        if np.sum(clu_sel)==0: continue # ，faiss
        cnt = Counter()
        for z in unlabeled_pseudo[clu_sel]:
            cnt[z] += 1
        # select minority from cnt
        lbl_freq = list(cnt.items())
        lbl_freq.sort(key=lambda x: x[1])
        clu_pseudo = unlabeled_pseudo[clu_sel]
        majlbl = lbl_freq[-1][0] # the majority label
        clu_majlbl[i] = majlbl
        majscore = unlabeled_pred[clu_sel][:, majlbl]
        dismaj = 1-majscore
        dis[clu_sel]= dismaj
        nonmaj_sel = clu_pseudo!=majlbl
        # clu_value[i] = np.mean(dismaj[nonmaj_sel]) # set i， 
        clu_value[i] =  np.sum(dismaj[nonmaj_sel])

    # pass 2: sample proportionanlly to clu_value
    cvsm = np.sum(clu_value)
    clu_nsample = [int(i/cvsm * n_sample) for i in clu_value]
    nmissing = n_sample - np.sum(clu_nsample)
    highclui=np.argsort(clu_value)[::-1][:nmissing]
    for i in highclui:
        clu_nsample[i]+=1

    for i in range(ncentroids):
        clu_sel = (I == i) # selector for current cluster
        topk = clu_nsample[i] # TODO: topk > clu_size, qqp 
        if topk<=0: continue
        majlbl = clu_majlbl[i]
        clu_pseudo = unlabeled_pseudo[clu_sel]
        majscore = unlabeled_pred[clu_sel][:, majlbl]
        nonmaj_sel = clu_pseudo!=majlbl
        nonmaj_idx = np.arange(len(clu_pseudo))[nonmaj_sel] # 
        npseudoerr = np.sum(nonmaj_sel)
        if npseudoerr > topk: # topk
            # random or entropy pick topk from w
            entp = entropy[clu_sel][nonmaj_sel]
            pi=  np.argsort(entp)[-topk:]
            picki = nonmaj_idx[pi]
        else:
            picki = np.argsort(majscore)[:topk]
        tmp = np.arange(len(I))[clu_sel][picki]
        sample_idx += tmp.tolist()

    dis_rank = np.argsort(dis)[::-1] # big 
    i=0
    # ，
    labeled=[False]*N
    for i in sample_idx:
        labeled[i]=True
    while len(sample_idx)<n_sample:
        j=dis_rank[i]
        if not labeled[j]:
            sample_idx.append(j)
            labeled[j]=True
        i+=1

    assert len(sample_idx) == n_sample
    for i, lbl in enumerate(labeled):
        if not lbl:
            save_idx.append(i)

    return sample_idx, save_idx, entropy, I
