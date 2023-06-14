"""
20221202: Created by cc
The real cal, no ablations or tricks 
"""


import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.nn.functional import normalize



def cal_emnlp21(args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred):
    """

    :param X_original: list of all data
    :param y_original: list of all labels
    :param labeled_inds: indices of current labeled/training examples
    :param discarded_inds: indices of examples that should not be considered for acquisition/annotation
    :param original_inds: indices of all data (this is a list of indices of the X_original list)
    :param results_dpool: dictionary with results from training/validation phase (for logits) of unlabeled set (pool)
    :param logits_dpool: logits for all examples in the pool
    :param train_dataset: the training set in the tensor format
    :param model: the fine-tuned model of the iteration
    :return:
    """
    """
    CAL (Contrastive Active Learning)
    Acquire data by choosing those with the largest KL divergence in the predictions between a candidate dpool input
     and its nearest neighbours in the training set.
     Our proposed approach includes:
     args.cls = True
     args.operator = "mean"
     the rest are False. We use them (True) in some experiments for ablation/analysis
     args.mean_emb = False
     args.mean_out = False
     args.bert_score = False 
     args.tfidf = False 
     args.reverse = False
     args.knn_lab = False
     args.ce = False
    :return:
    """
    # Use representations of current fine-tuned model *CAL*

    distances = None
    #####################################################
    # Contrastive Active Learning (CAL)
    #####################################################
    args_num_nei=10
    args_operator = "mean"
    args_ce=False
    neigh = KNeighborsClassifier(n_neighbors=args_num_nei)
    neigh.fit(X=train_feat, y=np.array(train_label))
    criterion = nn.KLDivLoss(reduction='none') if not args_ce else nn.CrossEntropyLoss()

    kl_scores = []
    distances = []#for loop
    for unlab_i, candidate in enumerate( zip(unlabeled_feat, unlabeled_pred)):
        # find indices of closesest "neighbours" in train set
        unlab_representation, unlab_logit = candidate
        distances_, neighbours = neigh.kneighbors(X=[candidate[0]], return_distance=True)
        neighbours = list(neighbours[0])
        distances.append(distances_[0])
        labeled_neighbours_labels = train_label[neighbours]
        # calculate score
        neigh_prob = train_pred[neighbours]
        # neigh_pseudo = np.argmax(neigh_prob, axis = 1)

        if args_ce:#ablation
            kl = np.array([criterion(unlab_logit.view(-1, args.n_labels ), label.view(-1)) for label in
                            labeled_neighbours_labels])
        else:
            candidate_log_prob = torch.Tensor( candidate[1] )
            kl = np.array([torch.sum(criterion(candidate_log_prob, torch.Tensor(n) ), dim=-1).numpy() for n in neigh_prob])
            # kl_scores.append(kl)
        # confidence masking
        if args_operator == "mean":
            kl_scores.append(kl.mean())
        elif args_operator == "max":
            kl_scores.append(kl.max())
        elif args_operator == "median":
            kl_scores.append(np.median(kl))

    distances = np.array([np.array(xi) for xi in distances]) # debug ，


    # select argmax
    # selected_inds = np.argpartition(kl_scores, -n_sample)[-n_sample:] # 
    iranks = np.argsort(kl_scores)[::-1]
    #############################################################################################################################
    iranks = list(iranks)

    return iranks, kl_scores
