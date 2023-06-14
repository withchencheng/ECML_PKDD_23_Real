import torch
import math


def entropy(logits, dim: int, keepdim: bool = False):
    # pred=torch.exp(logits) 
    return -torch.sum((torch.exp(logits) * logits).double(), dim=dim, keepdim=keepdim)


def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.

    We pass in logits.
    """
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])


def mutual_information(logits_B_K_C):
    """
    :param logits_B_K_C: list of tensors with logits after MC dropout.
    - B: |Dpool|
    - K: number of MC samples
    - C: number of classes
    :return:
    """

    sample_entropies_B_K = entropy(logits_B_K_C, dim=-1)
    entropy_mean_B = torch.mean(sample_entropies_B_K, dim=1)

    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C, dim=-1)

    mutual_info_B = mean_entropy_B - entropy_mean_B
    return mutual_info_B


def icml17bald(logits):
    """
    2011 Bayesian Active Learning by Disagreement (BALD).
    paper: https://arxiv.org/abs/1112.5745
    :param prob_dist:
    :return:
    """
    # # my way
    # # entropy
    # assert type(prob_dist) == list
    # mean_MC_prob_dist = torch.mean(torch.stack(prob_dist), 0)     # mean of N MC stochastic passes
    # prbslogs = mean_MC_prob_dist * torch.log2(mean_MC_prob_dist)  # p logp
    # numerator = 0 - torch.sum(prbslogs, dim=1)                    # -sum p logp
    # denominator = math.log2(mean_MC_prob_dist.size(1))            # class normalisation
    #
    # entropy = numerator / denominator
    #
    # # expectation of entropy
    # prob_dist_tensor = torch.stack(prob_dist, dim=-1)                                  # of shape (#samples, C, N)
    # classes_sum = torch.sum(prob_dist_tensor * torch.log2(prob_dist_tensor), dim=-1)   # of shape (#samples, C)
    # MC_sum = torch.sum(classes_sum, -1)                                                # of shape (#samples)
    #
    # expectation_of_entropy = MC_sum
    #
    # mutual_information_ = entropy + expectation_of_entropy

    # bb way
    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)
    bald_scores = mutual_information(logits_B_K_C)
    return bald_scores.cpu().numpy()
