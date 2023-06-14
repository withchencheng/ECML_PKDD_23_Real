"""
Code from https://github.com/forest-snow/alps
"""

import numpy as np
import torch

from real.baselines.alps.cluster import kmeans, kmeans_pp, badge_clu, kcenter
from real.baselines.alps.sample import get_scores_or_vectors, badge_gradient1

# logger = logging.getLogger(__name__)


def cluster_method(al_method):
    """Given the [al_method] method for active learning, return clustering function [f]
     and [condition], boolean that indicates whether sampling
    is conditioned on prior iterations
    
    
    """
    if "KM" in al_method:
        f = kmeans
        condition = False
    elif "KP" in al_method:
        f = kmeans_pp
        condition = True
    elif "FF" in al_method:
        f = kcenter
        condition = True # ------------
    elif "badge" in al_method:
        f = badge_clu
        condition = False
    elif "alps" in al_method:
        f = kmeans
        condition = False
    else:
        #  [al_method] is not cluster-based strategy
        f = None
        condition = None
    return f, condition

def acquire(pool, sampled, args, model, tokenizer):
    """Sample data from     unlabeled data [pool].
    The sampling method may need [args], [model], [tokenizer], or previously
    [sampled] data.
    
    Args:
      pool: The dataset
      sampled: idx for sampled in pool.
    
    """
    scores_or_vectors = get_scores_or_vectors(pool, args, model, tokenizer)
    clustering_fn, condition = cluster_method(args.al_method)
    unsampled = np.delete(torch.arange(len(pool)), sampled)
    # cluster-based sampling method like BADGE and ALPS
    vectors = torch.nn.functional.normalize(scores_or_vectors)
    centers = sampled.tolist()
    if not condition:
        #  badge/alps 
        # do not condition on previously chosen points,
        queries_unsampled = clustering_fn(
            vectors[unsampled], k = args.sample_labels
        )
        # add new samples to previously sampled list
        sample_idx = centers + (unsampled[queries_unsampled]).tolist()
    else:
        sample_idx = clustering_fn(
            vectors,
            k = args.sample_labels,
            centers = centers
        )
    sample_idx = torch.LongTensor(sample_idx)
    assert len(sample_idx) == len(sample_idx.unique()), "Duplicates found in sampling"
    assert len(sample_idx) > 0, "Sampling method sampled no queries."
    return sample_idx.cpu().numpy()


def iclr20badge(args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred, multi=1):
    """
    badgeKmeans++，，。
    ，。

    Args:
      multi=4  4 candidate  

    Returns:
      iranks, score
       
       center 
    """
    N = unlabeled_pred.shape[0]
    grads = badge_gradient1(unlabeled_pred, unlabeled_feat, args.n_labels)
    # cluster-based sampling method like BADGE and ALPS
    vectors = torch.nn.functional.normalize(grads )
    #  badge/alps 
    # do not condition on previously chosen points,
    queries_unsampled = badge_clu(
        vectors, k = args.sample_labels * multi
    )
    # add new samples to previously sampled list
    sample_idx_m = queries_unsampled
    sample_idx_m = torch.LongTensor(sample_idx_m)
    assert len(sample_idx_m) == len(sample_idx_m.unique()), "Duplicates found in sampling"
    assert len(sample_idx_m) > 0, "Sampling method sampled no queries."
    return sample_idx_m.tolist()

    # return iranks, entropy


def plm_KM(args,n_sample, train_feat, train_pred, train_label, unlabeled_feat, unlabeled_pred, multi=1):
    """
    pretrain language model embedding & k-means
      n_sample， error

    Args:
      multi=4  4 candidate  

    Returns:
      iranks, score
       
       center 
    """
    N = unlabeled_feat.shape[0]
    # cluster-based sampling method like BADGE and ALPS
    vectors = unlabeled_feat
    #  badge/alps 
    # do not condition on previously chosen points,
    queries_unsampled = kmeans(
        vectors, k = args.sample_labels * multi
    )
    # add new samples to previously sampled list
    sample_idx_m = queries_unsampled
    sample_idx_m = torch.LongTensor(sample_idx_m)
    assert len(sample_idx_m) == len(sample_idx_m.unique()), "Duplicates found in sampling"
    assert len(sample_idx_m) > 0, "Sampling method sampled no queries."
    return sample_idx_m.tolist()

    # return iranks, entropy


# def main():
#     args = setup.get_args()
#     setup.set_seed(args)
#
#     print(args)
#
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#
#     # Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
#     )

#     args.task_name = args.task_name.lower()
#     if args.task_name not in processors:
#         raise ValueError("Task not found: %s" % (args.task_name))
#     # first, get already sampled points
#     sampled_file = os.path.join(args.output_dir, 'sampled.pt')
#     if os.path.isfile(sampled_file):
#         sampled = torch.load(sampled_file)
#     else:
#         sampled = torch.LongTensor([])
#
#     # decide which model to load based on sampling method
#     args.head = sampling_to_head(args.sampling)
#     if args.head == "lm":
#         # load pre-trained model
#         args.model_name_or_path = args.base_model
#     model, tokenizer, _, _= setup.load_model(args)
#
#
#     dataset = train.load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
#
#     logger.info(f"Already sampled {len(sampled)} examples")
#     sampled = acquire(dataset, sampled, args, model, tokenizer)
#
#
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#
#     torch.save(sampled, os.path.join(args.output_dir, 'sampled.pt'))
#     logger.info(f"Sampled {len(sampled)} examples")
#     return len(sampled)
#
# if __name__ == "__main__":
#     main()
