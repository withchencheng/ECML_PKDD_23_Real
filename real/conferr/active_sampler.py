"""

1_0: confident error

uncertain_err_1_0

"""
import pickle
import torch
import numpy as np
from collections import Counter
from torch.utils.data.sampler import Sampler
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import pairwise_distances
import faiss
from dataclasses import dataclass
from real.baselines.emnlp21cal import cal_emnlp21
from real.baselines.icml17bald import icml17bald
from real.baselines.naacl22actune import naacl22actune 
from typing import List
from real.conferr.pseudo_err import pseudo_err_1_2, pseudo_err_1_3, pseudo_err_1_4, pseudo_err_1_4_1, pseudo_err_1_4_2
from real.conferr.utils import compute_entropy
from real.baselines.alps.active import iclr20badge, plm_KM
import random


class SubsetSampler(Sampler):
    r"""Samples elements seqentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)



@dataclass
class ALRoundInfo:
    spi: List[int] #raw id in unlabeled
    uemb: np.ndarray # unlabeled [CLS] embedding
    lemb: np.ndarray # labeled embedding
    uentropy: List[np.ndarray] # unlabeled entropy; added AL method-specific score;
    upred: np.ndarray 
    lpred:  np.ndarray # pred on labeled instances


class Active_sampler(object):
    
    def __init__(self, args, train_dataset, unlabeled_dataset, seed=0):
        self.args = args
        self.npr = np.random.RandomState(seed)
        self.train_dataset = train_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.al_info=[] # index is AL round

    def convert_tensor_to_dataset(self, tensor, prediction = None):
        if prediction is None:
            return TensorDataset(tensor[0],tensor[1], tensor[2],tensor[3],tensor[4],)
        else:
            prediction = torch.FloatTensor(prediction)
            # print(tensor[0].shape,tensor[1].shape, tensor[2].shape,tensor[3].shape,tensor[4].shape, prediction.shape)
            return TensorDataset(tensor[0],tensor[1], tensor[2],tensor[3],tensor[4], prediction)

    def init_trainset(self):
        N=len(self.unlabeled_dataset)
        rand_idx = np.random.permutation(N)
        sample_idx = rand_idx[0:self.args.warm_size]
        save_idx = rand_idx[self.args.warm_size:]
        sample_idx.sort()
        save_idx.sort()

        save_path = self.args.output_dir+'warm_up_ids.pkl'
        pickle.dump(sample_idx, open(save_path, 'wb'))
        self.train_dataset = self.convert_tensor_to_dataset(self.unlabeled_dataset[sample_idx])
        unlabeled_dataset = self.convert_tensor_to_dataset(self.unlabeled_dataset[save_idx])
        self.unlabeled_dataset = unlabeled_dataset
        return self.train_dataset ,  self.unlabeled_dataset 


    def sample(self, al_method, train_pred, train_feat, train_label, unlabeled_pred, unlabeled_feat, unlabeled_label, n_sample, round = 1):
        """
        sample from unlabeled.
        Return sampled_Info
        """
        print(f"Active sampling: {al_method}， Samping {n_sample} data")
        self.train_pred = train_pred
        self.train_feat = train_feat
        self.train_label = train_label
        self.unlabeled_pred = unlabeled_pred
        self.unlabeled_feat = unlabeled_feat
        self.unlabeled_label = unlabeled_label
        if self.args.save_sample:
            self.unlabeled_pseudo = np.argmax(unlabeled_pred, axis = -1)
            if unlabeled_label is not None:
                self.unlabeled_correct = (self.unlabeled_pseudo == unlabeled_label).astype(int)
            else:
                self.unlabeled_correct = None

        al_method2fn={
            'conferr_1.0': self.get_conferr_1_0,
            'uncterr_1.0': self.get_uncterr_1_0,
            'uncterr_1.0.1': self.get_uncterr_1_0_1,#entropy threshold for uncertainty, control only the upper bound 
            'uncterr_1.0.2': self.get_uncterr_1_0_2,#entropy threshold for uncertainty, control only the lower bound
            'uncterr_1.0.3': self.get_uncterr_1_0_3,#threshold for uncertainty, control by multiples of nsample  [1, #unlabel_pool/nsample]
            'conferr_1.1': self.get_conferr_1_1,
            'conferr_1.1.1': self.get_conferr_1_1_1,
            'uncterr_1.1': self.get_uncterr_1_1,
            'pseudo_err_1.2': self.get_pseudo_err_1_2,
            'pseudo_err_1.3': self.get_pseudo_err_1_3,
            'pseudo_err_1.4': self.get_pseudo_err_1_4,
            'pseudo_err_1.4.1': self.get_pseudo_err_1_4_1,
            'pseudo_err_1.4.2': self.get_pseudo_err_1_4_2,
            'entropy': self.get_max_entropy,
            'entropy_div': self.get_entropy_div, 
            'random': self.get_random,
            'emnlp21cal': self.get_emnlp21cal,
            'icml17bald': self.get_icml17bald, # 
            'naacl22actune': self.get_naacl22actune, 
            'naacl22actune_1.0.3': self.get_naacl22actune_1_0_3, 
            'emnlp21cal_1.0.3': self.get_emnlp21cal_1_0_3, 
            'iclr20badge': self.get_iclr20badge, 
            'emnlp20alps': self.get_emnlp20alps, 
            'plm_KM': self.get_plm_KM, 
        }
        alfn=al_method2fn[al_method]
        ret = alfn(n_sample, unlabeled_pred, unlabeled_feat)
        if 'emnlp21cal' in al_method: #
            sample_idx, save_idx, entropy, uscores = ret
            uentropy=[entropy, uscores]
        elif 'pseudo_err' in al_method: # in ['pseudo_err_1.2','pseudo_err_1.3', 'pseudo_err_1.4'] 
            sample_idx, save_idx, entropy, clu_ids = ret
            uentropy=[entropy, clu_ids]
        else:
            sample_idx, save_idx, entropy = ret
            uentropy=[entropy, None]
        
        sample_idx.sort()
        save_idx.sort()

        sample_dataset = self.convert_tensor_to_dataset(self.unlabeled_dataset[sample_idx])
        unlabeled_dataset = self.convert_tensor_to_dataset(self.unlabeled_dataset[save_idx])
        if self.args.save_sample:
            self._get_sample_info(sample_dataset, uentropy)#

        train_dataset = ConcatDataset([self.train_dataset, sample_dataset])
        self.train_dataset = train_dataset
        
        self.unlabeled_dataset = unlabeled_dataset
        return self.train_dataset, self.unlabeled_dataset
    

    def _get_sample_info(self, sample_dataset, uentropy):
        spi=[ins[4].item() for ins in sample_dataset]
        pad_token=0
        ari = ALRoundInfo(spi, self.unlabeled_feat, self.train_feat,  uentropy
              , self.unlabeled_pred, self.train_pred
              )
        self.al_info.append(ari)
        return ari
    
    def save_sample_info(self):
        """Call me after all AL finishes"""
        save_path = self.args.output_dir+'al_info.pkl'
        print('Save sample info to ', save_path)
        pickle.dump(self.al_info, open(save_path, 'wb'))


    def get_random(self,  n_sample, unlabeled_pred, unlabeled_feat):
        entropy = compute_entropy(unlabeled_pred)
        len_unlabel = unlabeled_pred.shape[0]
        rand_idx = np.random.permutation(len_unlabel)
        sample_idx = rand_idx[0:n_sample]
        save_idx = rand_idx[n_sample:]
        sample_idx = list(sample_idx)
        save_idx = list(save_idx)
        return sample_idx, save_idx, entropy

    def get_max_entropy(self, n_sample, unlabeled_pred, unlabeled_feat):
        entropy = compute_entropy(unlabeled_pred)
        idx = np.argsort(entropy)[::-1]
        sample_idx = idx[0:n_sample]
        save_idx = idx[n_sample:]
        sample_idx = list(sample_idx)
        save_idx = list(save_idx)
        return sample_idx, save_idx, entropy

    def get_conferr_1_0(self, n_sample, unlabeled_pred, unlabeled_feat):
        labeled = [False for i in range(unlabeled_pred.shape[0])]

        entropy = compute_entropy(unlabeled_pred)
        idx = np.argsort(entropy)# 
        n=len(idx)
        sample_idx, save_idx = [], []
        for i in idx:
            if self.unlabeled_correct[i]: continue
            if len(sample_idx)<n_sample:
                sample_idx.append(i)
                labeled[i]=True
            else:
                break
        # edge case: almost all right
        i=n-1
        while  len(sample_idx)<n_sample:
            j=idx[i]
            if not labeled[j]: 
                sample_idx.append(j)
                labeled[j]=True
            i -= 1

        for i, lbl in enumerate(labeled) :
            if not lbl:
                save_idx.append(i)
        return sample_idx, save_idx, entropy


    def get_uncterr_1_0(self, n_sample, unlabeled_pred, unlabeled_feat):
        labeled = [False for i in range(unlabeled_pred.shape[0])]
        entropy = compute_entropy(unlabeled_pred)
        idx = np.argsort(entropy)[::-1] # uncertain
        sample_idx, save_idx = [], []
        for i in idx:
            if self.unlabeled_correct[i]: continue
            if len(sample_idx)<n_sample:
                sample_idx.append(i)
                labeled[i]=True
            else:
                break
        # edge case: almost all right
        i=0
        while len(sample_idx)<n_sample:
            j=idx[i]
            if not labeled[j]: 
                sample_idx.append(j)
                labeled[j]=True
            i += 1
        assert len(sample_idx)==n_sample

        for i, lbl in enumerate(labeled) :
            if not lbl:
                save_idx.append(i)
        return sample_idx, save_idx, entropy


    def _err_1_0_3(self, n_sample, iranks):
        """
        iranks
            lowbound =  self.args.unctth * n_sample # rank lower bound
        Args:
          iranks : al method  unlabeled instances  argsort， top1
        Returns:
          sample_idx, save_idx
        """
        n=len(iranks)
        labeled = [False for i in range(n)]
        sample_idx, save_idx = [], []
        lowbound =  self.args.unctth * n_sample # rank lower bound

        i=0
        while len(sample_idx)<n_sample and i<n:
            j=iranks[i]
            i += 1
            if i>=lowbound: break
            if self.unlabeled_correct[j]: continue
            sample_idx.append(j)
            labeled[j]=True

        # edge case: almost all right
        i=0
        while  len(sample_idx)<n_sample:
            j=iranks[i]
            if not labeled[j]:
                sample_idx.append(j)
                labeled[j]=True
            i += 1

        for i, lbl in enumerate(labeled) :
            if not lbl:
                save_idx.append(i)

        assert len(sample_idx)==n_sample
        return sample_idx, save_idx


    def get_uncterr_1_0_1(self, n_sample, unlabeled_pred, unlabeled_feat):
        """ uncertainty-error considering uncertainty threshold. """
        labeled = [False for i in range(unlabeled_pred.shape[0])]
        musthave = int(n_sample*self.args.unctth)
        entropy = compute_entropy(unlabeled_pred)
        idx = np.argsort(entropy)[::-1] # uncertain 
        sample_idx, save_idx = [], []
        sample_idx = idx[:musthave].tolist()
        for i in sample_idx:
            labeled[i]=True
        for i in idx[musthave:]:
            if self.unlabeled_correct[i]: continue
            if len(sample_idx)<n_sample:
                sample_idx.append(i)
                labeled[i]=True
            else:
                break
        # edge case: almost all right
        i=musthave
        while  len(sample_idx)<n_sample:
            j=idx[i]
            if not labeled[j]:
                sample_idx.append(j)
                labeled[j]=True
            i += 1

        for i, lbl in enumerate(labeled) :
            if not lbl:
                save_idx.append(i)
        return sample_idx, save_idx, entropy


    def get_uncterr_1_0_2(self, n_sample, unlabeled_pred, unlabeled_feat):
        """ Uncertainty-error considering uncertainty threshold.
        Control the lower bound of sampled uncertainty.
        """
        n=unlabeled_pred.shape[0]
        labeled = [False for i in range(n)]
        unctth = self.args.unctth
        entropy = compute_entropy(unlabeled_pred)
        idx = np.argsort(entropy)[::-1] # uncertain 
        sample_idx, save_idx = [], []

        i=0
        while len(sample_idx)<n_sample and i<n:
            j=idx[i]
            i += 1
            if self.unlabeled_correct[j]: continue
            if entropy[j]>unctth:
                sample_idx.append(j)
                labeled[j]=True
            else:
                break

        # edge case: almost all right
        i=0
        while  len(sample_idx)<n_sample:
            j=idx[i]
            if not labeled[j]:
                sample_idx.append(j)
                labeled[j]=True
            i += 1

        for i, lbl in enumerate(labeled) :
            if not lbl:
                save_idx.append(i)
        return sample_idx, save_idx, entropy


    def get_uncterr_1_0_3(self, n_sample, unlabeled_pred, unlabeled_feat):
        """ Uncertainty-error considering uncertainty threshold.
        Control the lower bound of sampled uncertainty.
        """
        n=unlabeled_pred.shape[0]
        labeled = [False for i in range(n)]
        entropy = compute_entropy(unlabeled_pred)
        idx = np.argsort(entropy)[::-1] # uncertain 
        sample_idx, save_idx = [], []
        lowbound =  self.args.unctth * n_sample # rank lower bound

        i=0
        while len(sample_idx)<n_sample and i<n:
            j=idx[i]
            i += 1
            if i>=lowbound: break
            if self.unlabeled_correct[j]: continue
            sample_idx.append(j)
            labeled[j]=True

        # edge case: almost all right
        i=0
        while  len(sample_idx)<n_sample:
            j=idx[i]
            if not labeled[j]:
                sample_idx.append(j)
                labeled[j]=True
            i += 1

        for i, lbl in enumerate(labeled) :
            if not lbl:
                save_idx.append(i)
        return sample_idx, save_idx, entropy


    def get_uncterr_1_1(self, n_sample, unlabeled_pred, unlabeled_feat):
        """
        Use pseudo error rather than real error.
        No accesss to unlabel_correct 
        Choose uncertain & pred_label being the minority in the clusters

        ncentroids = 80，uncertainty
        20221121: ？
        """
        n=unlabeled_pred.shape[0]
        clusize=10
        ncentroids = self.args.ncentroids
        entropy = compute_entropy(unlabeled_pred)
        entp_rank = np.argsort(entropy)[::-1] # uncertain 
        sample_idx, save_idx = [], []

        d = unlabeled_feat.shape[-1]
        unlabeled_pseudo = np.argmax(unlabeled_pred, axis = -1)
        kmeans = faiss.Clustering(int(d), ncentroids)
        index = faiss.IndexFlatL2(d)
        kmeans.train(unlabeled_feat, index)
        centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
        index.add(centroid)
        D, I = index.search(unlabeled_feat, 1)
        I = I.flatten() # list of cluster assignment for all unlabeled ins
        scores = []
        indexes = [] # list of sorted idx for each cluster
        topk = n_sample//ncentroids
        for i in range(ncentroids):
            idx = (I == i)
            cnt = Counter()
            # calculate the mean entropy of samples
            # mean_entropy = np.mean(entropy[idx])
            for z in unlabeled_pseudo[idx]:
                cnt[z] += 1
            # select minority from cnt
            lbl_freq = list(cnt.items())
            lbl_freq.sort(key=lambda x: x[1])
            minority = lbl_freq[0][0]
            # value = mean_entropy + beta * class_entropy
            sorted_idx = np.argsort(entropy[idx])[::-1]   #clusteridx
            idxs = np.arange(len(I))[idx][sorted_idx]  # unlabeled poolidx        
            tmp=[]
            j=0
            while j<len(idxs) and len(tmp)<topk: # assumecluster，topkminority
                i1=idxs[j]
                if unlabeled_pseudo[i1] == minority:
                    tmp.append(i1)
                j+=1
            sample_idx += tmp

        labeled=[False]*n
        for i in sample_idx:
            labeled[i]=True

        i=0
        while len(sample_idx)<n_sample:
            j=entp_rank[i]
            if not labeled[j]:
                sample_idx.append(j)
                labeled[j]=True
            i+=1

        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)

        return sample_idx, save_idx, entropy

        
    def get_pseudo_err_1_2(self, n_sample, unlabeled_pred, unlabeled_feat):
        """
        Use pseudo error rather than real error.
        No accesss to unlabel_correct 
        Choose uncertain & pred_label smallest on majority label in the clusters
        Select equal number of samples per cluster

        ncentroids = 80，uncertainty
        20221121: ？
        """
        sample_idx, save_idx, entropy, clu_ids = pseudo_err_1_2(self.args, n_sample, self.train_feat, self.train_pred, self.train_label, unlabeled_feat, unlabeled_pred)
        return sample_idx, save_idx, entropy, clu_ids


    def get_pseudo_err_1_3(self, n_sample, unlabeled_pred, unlabeled_feat):
        """
        Use pseudo error rather than real error.
        No accesss to unlabel_correct 
        Choose uncertain & pred_label smallest on majority label in the clusters
        Rank over the whole unlabeled pool

        ncentroids = 80，uncertainty
        20221121: ？
        """
        sample_idx, save_idx, entropy, clu_ids = pseudo_err_1_3(self.args, n_sample, self.train_feat, self.train_pred, self.train_label, unlabeled_feat, unlabeled_pred)
        return sample_idx, save_idx, entropy, clu_ids


    def get_pseudo_err_1_4(self, n_sample, unlabeled_pred, unlabeled_feat):
        """
        Use pseudo error rather than real error.
        No accesss to unlabel_correct 
        Choose uncertain & pred_label smallest on majority label in the clusters

        ncentroids = 80，uncertainty
        20221121: ？
        """
        sample_idx, save_idx, entropy, clu_ids = pseudo_err_1_4(self.args, n_sample, self.train_feat, self.train_pred, self.train_label, unlabeled_feat, unlabeled_pred)
        return sample_idx, save_idx, entropy, clu_ids


    def get_pseudo_err_1_4_1(self, n_sample, unlabeled_pred, unlabeled_feat):
        sample_idx, save_idx, entropy, clu_ids = pseudo_err_1_4_1(self.args, n_sample, self.train_feat, self.train_pred, self.train_label, unlabeled_feat, unlabeled_pred)
        return sample_idx, save_idx, entropy, clu_ids

    def get_pseudo_err_1_4_2(self, n_sample, unlabeled_pred, unlabeled_feat):
        sample_idx, save_idx, entropy, clu_ids = pseudo_err_1_4_2(self.args, n_sample, self.train_feat, self.train_pred, self.train_label, unlabeled_feat, unlabeled_pred)
        return sample_idx, save_idx, entropy, clu_ids


    def get_conferr_1_1(self, n_sample, unlabeled_pred, unlabeled_feat):
        """
        use pseudo error rather than real error
        no accesss to unlabel_correct 
        choose confident & pred_label being the minority in the clusters

        ncentroids = 80，uncertainty
        """
        n=unlabeled_pred.shape[0]
        ncentroids = self.args.ncentroids
        entropy = compute_entropy(unlabeled_pred)
        entp_rank = np.argsort(entropy) # confident 
        sample_idx, save_idx = [], []

        d = unlabeled_feat.shape[-1]
        kmeans = faiss.Clustering(int(d), ncentroids)
        index = faiss.IndexFlatL2(d)
        kmeans.train(unlabeled_feat, index)
        centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
        index.add(centroid)
        D, I = index.search(unlabeled_feat, 1)
        I = I.flatten() # list of cluster assignment for all unlabeled ins
        topk = n_sample//ncentroids
        for i in range(ncentroids):
            idx = (I == i)
            cnt = Counter()
            # calculate the mean entropy of samples
            # mean_entropy = np.mean(entropy[idx])
            for z in self.unlabeled_pseudo[idx]:
                cnt[z] += 1
            # select minority from cnt
            lbl_freq = list(cnt.items())
            lbl_freq.sort(key=lambda x: x[1])
            minority = lbl_freq[0][0]
            # value = mean_entropy + beta * class_entropy
            sorted_idx = np.argsort(entropy[idx]) 
            idxs = np.arange(len(I))[idx][sorted_idx]                     
            tmp=[]
            j=0
            while j<len(idxs) and len(tmp)<topk: # assumecluster，topkminority
                i1=idxs[j]
                if self.unlabeled_pseudo[i1] == minority:
                    tmp.append(i1)
                j+=1
            sample_idx += tmp
        labeled=[False]*n
        for i in sample_idx:
            labeled[i]=True

        i=n-1
        while len(sample_idx)<n_sample:
            j=entp_rank[i]
            if not labeled[j]:
                sample_idx.append(j)
                labeled[j]=True
            i-=1

        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)

        return sample_idx, save_idx, entropy


    def get_conferr_1_1_1(self, n_sample, unlabeled_pred, unlabeled_feat):
        """
        use pseudo error rather than real error
        no accesss to unlabel_correct 
        choose top uncertain ones, and then choose confident & pred_label being the minority in the clusters
        """
        n=unlabeled_pred.shape[0]
        ncentroids = 80
        entropy = compute_entropy(unlabeled_pred)
        entp_rank = np.argsort(entropy) # confident 
        sample_idx, save_idx = [], []

        d = unlabeled_feat.shape[-1]
        kmeans = faiss.Clustering(int(d), ncentroids)
        index = faiss.IndexFlatL2(d)
        kmeans.train(unlabeled_feat, index)
        centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
        index.add(centroid)
        D, I = index.search(unlabeled_feat, 1)
        I = I.flatten() # list of cluster assignment for all unlabeled ins
        topk = n_sample//ncentroids
        for i in range(ncentroids):
            idx = (I == i)
            cnt = Counter()
            # calculate the mean entropy of samples
            # mean_entropy = np.mean(entropy[idx])
            for z in self.unlabeled_pseudo[idx]:
                cnt[z] += 1
            # select minority from cnt
            lbl_freq = list(cnt.items())
            lbl_freq.sort(key=lambda x: x[1])
            minority = lbl_freq[0][0]
            # value = mean_entropy + beta * class_entropy
            sorted_idx = np.argsort(entropy[idx]) 
            idxs = np.arange(len(I))[idx][sorted_idx]                     

            tmp=[]
            j=0
            while j<len(idxs) and len(tmp)<topk: # assumecluster，topkminority
                i1=idxs[j]
                if self.unlabeled_pseudo[i1] == minority:
                    tmp.append(i1)
                j+=1
            sample_idx += tmp
        labeled=[False]*n
        for i in sample_idx:
            labeled[i]=True

        i=n-1
        while len(sample_idx)<n_sample:
            j=entp_rank[i]
            if not labeled[j]:
                sample_idx.append(j)
                labeled[j]=True
            i-=1

        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)

        return sample_idx, save_idx, entropy

    def get_entropy_div(self, n_sample, unlabeled_pred, unlabeled_feat):
        """
        use pseudo error rather than real error
        no accesss to unlabel_correct 
        choose uncertain & pred_label being the minority in the clusters
        """
        n=unlabeled_pred.shape[0]
        clusize=10
        ncentroids = 80
        entropy = compute_entropy(unlabeled_pred)
        entp_rank = np.argsort(entropy)[::-1] # confident 
        sample_idx, save_idx = [], []

        d = unlabeled_feat.shape[-1]
        kmeans = faiss.Clustering(int(d), ncentroids)
        index = faiss.IndexFlatL2(d)
        kmeans.train(unlabeled_feat, index)
        centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
        index.add(centroid)
        D, I = index.search(unlabeled_feat, 1)
        I = I.flatten() # list of cluster assignment for all unlabeled ins
        scores = []
        indexes = [] # list of sorted idx for each cluster
        topk = n_sample//ncentroids
        for i in range(ncentroids):
            idx = (I == i)
            # value = mean_entropy + beta * class_entropy
            sorted_idx = np.argsort(entropy[idx])[::-1] 
            idxs = np.arange(len(I))[idx][sorted_idx]                     
            tmp=[]
            j=0
            while j<len(idxs) and len(tmp)<topk: # assumecluster，topkminority
                i1=idxs[j]
                tmp.append(i1)
                j+=1
            sample_idx += tmp
        labeled=[False]*n
        for i in sample_idx:
            labeled[i]=True

        i=0
        while len(sample_idx)<n_sample:
            if not labeled[i]:
                sample_idx.append(i)
                labeled[i]=True
            i+=1

        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)

        return sample_idx, save_idx, entropy

    def get_icml17bald(self, n_sample, unlabeled_pred, unlabeled_feat):
        logits = unlabeled_pred # 
        n=unlabeled_pred[0].shape[0] # !!CAUTION!! bald  ！！
        entropy = None
        sample_idx, save_idx = [], []
        #args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred
        uncertainty_scores=icml17bald(logits)
        sample_idx = np.argpartition(uncertainty_scores, -n_sample)[-n_sample:]
        sample_idx = list(sample_idx)
        labeled=[False]*n
        for i in sample_idx:
            labeled[i]=True
        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)
        assert len(sample_idx)==n_sample
        return sample_idx, save_idx, uncertainty_scores


    def get_naacl22actune(self, n_sample, unlabeled_pred, unlabeled_feat):
        n=unlabeled_pred.shape[0] 
        sample_idx, save_idx = [], []
        #args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred
        iranks, entropy = naacl22actune(self.args, n_sample, unlabeled_feat, unlabeled_pred)
        sample_idx = iranks[:n_sample]
        labeled=[False]*n
        for i in sample_idx:
            labeled[i]=True
        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)

        assert len(sample_idx)==n_sample
        return sample_idx, save_idx, entropy


    def get_emnlp21cal(self, n_sample, unlabeled_pred, unlabeled_feat):
        n=unlabeled_pred.shape[0]
        entropy = compute_entropy(unlabeled_pred)
        sample_idx, save_idx = [], []
        #args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred
        iranks, kl_scores=cal_emnlp21(self.args, n_sample, self.train_feat, self.train_pred, self.train_label, unlabeled_feat, unlabeled_pred )
        sample_idx = iranks[:n_sample]

        labeled=[False]*n
        for i in sample_idx:
            labeled[i]=True
        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)
        return sample_idx, save_idx, entropy, kl_scores

    def get_iclr20badge(self, n_sample, unlabeled_pred, unlabeled_feat):
        n=unlabeled_pred.shape[0]
        entropy = compute_entropy(unlabeled_pred)
        sample_idx, save_idx = [], []
        #args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred
        iranks =iclr20badge(self.args, n_sample, self.train_feat, self.train_pred, self.train_label, unlabeled_feat, unlabeled_pred, multi=1 )
        sample_idx = iranks[:n_sample]

        labeled=[False]*n
        for i in sample_idx:
            labeled[i]=True
        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)
        return sample_idx, save_idx, entropy


    def get_plm_KM(self, n_sample, unlabeled_pred, unlabeled_feat):
        # pretrain language model k-means
        n=unlabeled_pred.shape[0]
        entropy = compute_entropy(unlabeled_pred)
        sample_idx, save_idx = [], []
        #args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred
        iranks = plm_KM(self.args, n_sample, self.train_feat, self.train_pred, self.train_label, unlabeled_feat, unlabeled_pred)
        sample_idx = iranks[:n_sample]

        labeled=[False]*n
        for i in sample_idx:
            labeled[i]=True
        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)
        return sample_idx, save_idx, entropy


    def get_emnlp20alps(self, n_sample, unlabeled_pred, unlabeled_feat):
        # pretrain language model k-means
        n=unlabeled_feat.shape[0]
        entropy = None
        sample_idx, save_idx = [], []
        #args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred
        iranks = plm_KM(self.args, n_sample, self.train_feat, self.train_pred, self.train_label, unlabeled_feat, unlabeled_pred) # ， PLM KM，，  unlabeled_feat  surprisal emb
        sample_idx = iranks[:n_sample]

        labeled=[False]*n
        for i in sample_idx:
            labeled[i]=True
        for i, lbl in enumerate(labeled):
            if not lbl:
                save_idx.append(i)
        return sample_idx, save_idx, entropy


# --------- Best recent baseline + ground truth error ----------------

    def get_naacl22actune_1_0_3(self, n_sample, unlabeled_pred, unlabeled_feat):
        n=unlabeled_pred.shape[0]
        sample_idx, save_idx = [], []
        #args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred
        iranks, entropy = naacl22actune(self.args, n_sample, unlabeled_feat, unlabeled_pred)

        sample_idx, save_idx = self._err_1_0_3(n_sample, iranks)
        return sample_idx, save_idx, entropy


    def get_emnlp21cal_1_0_3(self, n_sample, unlabeled_pred, unlabeled_feat):
        n=unlabeled_pred.shape[0]
        entropy = compute_entropy(unlabeled_pred)
        sample_idx, save_idx = [], []
        #args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred
        iranks, kl_scores =cal_emnlp21(self.args, n_sample, self.train_feat, self.train_pred, self.train_label, unlabeled_feat, unlabeled_pred )

        sample_idx, save_idx = self._err_1_0_3(n_sample, iranks)
        return sample_idx, save_idx, entropy, kl_scores