
## Version meaning

            'pseudo_err_1_4_1': The Real method, random sample pseudo errors within a cluster
            'pseudo_err_1_4': Ablation, rank by errornous probability within a cluster
            'pseudo_err_1_4_2':  Ablation, combine with entropy

            'conferr_1.0': confident error, know gold label before active sampling, cheated
            'uncterr_1.0': uncertain error, know gold label before active sampling, cheated
            'uncterr_1.0.1': control threshold for maintain top uncertain ones, 不像uncterr_1.0 只选错的 而不管 uncertainty
            'uncterr_1.0.2': control the lower bound for uncertainty, 不像uncterr_1.0 只选错的 而不管 uncertainty
            'uncterr_1.0.3': threshold for uncertainty, control by multiples of nsample  [1, #unlabel_pool/nsample]
            'conferr_1.1': confident error, pseudo error
            'uncterr_1.1': uncertain error, pseudo error
            'pseudo_err_1.2': 先聚类，再找 majority class 所属概率最低的样本
            'uncterr_1.1.1': threshold for maintain top uncertain ones.  uncertain error, pseudo error
            'conferr_1.1.1': threshold for maintain top uncertain ones.   confident error, pseudo error
            'entropy': ,
            'entropy_div': select top uncertain ones within each cluster
            'random': self.get_random,
            'emnlp21cal': self.get_emnlp21cal,
            'icml17bald': self.get_icml17bald, # 用unlabeled_pred 冒充 mc logits. 并且n=unlabeled_pred[0].shape[0] # !!CAUTION!!
            'naacl22actune': self.get_naacl22actune, 
            'naacl22actune_1.0.3':  threshold for the ranks to be sampled, control by multiples of nsample  [1, #unlabel_pool/nsample]
            'emnlp21cal_1.0.3':  threshold for the ranks to be sampled, control by multiples of nsample  [1, #unlabel_pool/nsample]
            'iclr20badge': self.get_iclr20badge, 
            'emnlp20alps': self.get_emnlp20alps, 
            'plm_KM': self.get_plm_KM,  # 


Note that Alps takes lots of GPU space to run.
