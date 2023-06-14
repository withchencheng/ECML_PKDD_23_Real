import os
import logging
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import  AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModelForSequenceClassification, RobertaForMaskedLM
import os
from real.conferr.active_sampler import Active_sampler
import json
from real.conferr.utils import set_seed, print_model_param
from real.baselines.alps.sample import mask_tokens, get_mlm_loss, batch_scores_or_vectors
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
    print('from torch.utils.tensorboard import SummaryWriter')
    pass
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)

def acc_and_f1(preds, labels):
    acc = (preds == labels).mean()
    maf1 = f1_score(labels, preds, average='macro')

    return {
        "acc": acc,
        "maf1": maf1
    }


class Trainer(object):
    def __init__(self, args, train_dataset = None, dev_dataset = None, test_dataset = None, unlabeled = None, \
                num_labels = 10):
        self.args = args
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.unlabeled = unlabeled

        self.num_labels = num_labels
        set_seed(args.seed)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        # self.model = RobertaForCls_Tri(args)
        # print_model_param(self.args, self.model)
        self.tb_writer = SummaryWriter(self.args.tsb_dir)
        self.active_sampler = Active_sampler(args = self.args, train_dataset = train_dataset, unlabeled_dataset = unlabeled)
        # Done model init. Fix randomness than sample batch
        set_seed(self.args.seed)
        self.train_dataset, self.unlabeled = self.active_sampler.init_trainset()  # ， self.unlabeled  100 
        # print(self.train_dataset[:4])

    def move_model_device(self):
        # GPU or CPU
        # self.device = f"cuda:{self.args.gpu}" if torch.cuda.is_available() and self.n_gpu > 0 else "cpu"
        self.device = self.args.device
        # if self.n_gpu > 1:
        #     self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
    
    ########################################## Load, save, init, close   ##########################################
    def load_model(self, path):
        logger.info(f"Loading from {path}!")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        # self.move_model_device()

    def save_model(self, path):
        logger.info("Saving model checkpoint to %s", path)
        torch.save(self.model.state_dict(), path)

    def reinit_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_name_or_path, num_labels=self.num_labels)
        self.move_model_device()

    def save_dataset(self, stage = 0):
        output_dir = os.path.join(
            self.args.output_dir, "dataset", "dataset-{}-{}-{}".format(self.args.model_type, self.args.al_method, stage))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.train_dataset, os.path.join(output_dir, 'train'))
        torch.save(self.unlabeled, os.path.join(output_dir, 'unlabeled'))


    def load_dataset(self, stage = 0):
        load_dir = os.path.join(
            self.args.output_dir, "dataset", "dataset-{}-{}-{}-{}".format(self.args.model_type, self.args.al_method, self.args.al_method, stage))
        if not os.path.exists(load_dir):
            # except:
            load_dir = os.path.join(
                self.args.output_dir, "dataset", "dataset-{}-{}-{}".format(self.args.model_type, self.args.al_method, stage))
        self.train_dataset = torch.load(os.path.join(load_dir, 'train'))
        self.unlabeled = torch.load(os.path.join(load_dir, 'unlabeled'))

    def save_result(self, stage = 0, acc = 0):
        output_dir = os.path.join(
            self.args.output_dir, "result", "result-{}-{}-{}-{}".format(self.args.model_type,self.args.al_method, self.args.al_method,   stage))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'acc.json') , 'w') as f:
            json.dump({"acc": acc, "stage": stage, "al_method": self.args.al_method, "model_type":self.args.model_type}, f)

    def close_clean(self):
        # close the trainer, save information, clean up
        if self.args.save_sample:
            self.active_sampler.save_sample_info()

    ########################################## Train   ##########################################
    def train(self, round):
        # n_sample=50;   100train_dataset
        # self.train_dataset, self.model, self.tb_writer, 
        set_seed(self.args.seed)
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        training_steps = int(self.args.num_train_epochs) * len(train_dataloader)

        num_warmup_steps = min(int(training_steps * 0.1), 100)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = training_steps)

        # Train!
        print("***** Running training *****")
        print("  Num train = %d", len(self.train_dataset))
        print("  Num Epochs = %d", self.args.num_train_epochs)
        print("  train batch size = %d", self.args.train_batch_size)
        print("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        print("  Total optimization steps = %d", self.args.max_steps)
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        save_best_path = f"{self.args.output_dir}best_tmp_pid{self.args.pid}.pth" 
        best_dev = -1
        stopall=False
        neval1epoch=self.args.neval1epoch
        eval_nstep=[int(i*(len(train_dataloader)/neval1epoch)) for i in range(1, neval1epoch)]
        eval_nstep.append(len(train_dataloader))
        runningw=0.9
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                step1=step+1
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3]
                          }
                outputs = self.model.forward(**inputs)
                loss = outputs[0]
                logits = outputs[1]
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                tr_loss = loss.item()
                if global_step==0:
                    oldloss=tr_loss 
                tr_loss = runningw*tr_loss + (1-runningw)*oldloss
                oldloss=tr_loss 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()

                global_step += 1
                if global_step % self.args.logging_steps == 0:# save train loss
                    self.tb_writer.add_scalar(f"Train_loss_round{round}", tr_loss, global_step)
                    print(f'Epoch {epoch}, step {step1}, global_step {global_step}, Train loss: {tr_loss}')
                if step1 in eval_nstep :# eval on dev
                    loss_dev, acc_dev, maf1_dev = self.evaluate('dev')
                    self.tb_writer.add_scalar(f"Dev_acc_round{round}", acc_dev, global_step)
                    self.tb_writer.add_scalar(f"Train_loss_round{round}", tr_loss, global_step)
                    print(f'Epoch {epoch}, step {step1}, global_step {global_step}, Train loss: {tr_loss} Dev Loss: {loss_dev}, Dev Acc: {acc_dev}, Macro-f1: {maf1_dev}')
                    if acc_dev > best_dev:
                        logger.info("Best model updated!")
                        torch.save(self.model.state_dict(), save_best_path) # official recommended
                        best_dev = acc_dev
                if global_step > self.args.max_steps:
                    stopall=True
                    break
                    # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    #     self.save_model(stage = round)
                
            loss_dev, acc_dev, maf1_dev  = self.evaluate('dev')
            print(f'Epoch {epoch}, , step {step1}, global_step {global_step},  Dev: Loss: {loss_dev}, Acc: {acc_dev}, Macro-f1: {maf1_dev}')
            self.tb_writer.add_scalar(f"Dev_acc_round{round}", acc_dev, global_step)
            if acc_dev > best_dev:
                logger.info("Best model updated!")
                torch.save(self.model.state_dict(), save_best_path)
                best_dev = acc_dev            
            if stopall:
                break
        
        if os.path.exists(save_best_path):
            self.model.load_state_dict(torch.load(save_best_path, map_location=self.device))
        else:
            logger.warning('Model not updated!!')
        loss_test, acc_test, maf1_test  = self.evaluate('test')
        print(f'Round {round}, Test: Loss: {loss_test}, Acc: {acc_test}, Macro-f1: {maf1_test}')
        self.tb_writer.add_scalar(f"Test_acc_{self.args.al_method}_seed{self.args.seed}", acc_test, round)
        if os.path.exists(save_best_path):
            os.remove(save_best_path)
        return global_step, tr_loss


    def sample(self, n_sample, n_unlabeled = 2048, round = 1):

        set_seed(self.args.seed)
        logger.info(f"Begin inference")
        if 'icml17bald' in self.args.al_method:
            train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label= self.inference_mc(layer = -1)
        elif 'alps' in self.args.al_method :
            train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label= self.inference_alps(layer = -1)# unlabeled_feat  surprisal embedding, None
        else:
            train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label= self.inference(layer = -1)# train_pred [100,num_labels] ; train_feat [100,768]; unlabeled_feat [40000,768]
        logger.info(f"Done inference")

        logger.info(f"Begin sample")
        new_train, new_unlabeled= self.active_sampler.sample(self.args.al_method,
            train_pred, train_feat, train_label,
            unlabeled_pred, unlabeled_feat, unlabeled_label,
             n_sample= n_sample, round = round)
        logger.info(f"Done sample")
        
        self.train_dataset = new_train
        self.unlabeled = new_unlabeled
        print(f"=======  train {len(new_train)}, unlabel {len(new_unlabeled)}  =========")
        # self.save_dataset(stage = n_sample)
        return

    def _inf1ds(self, ds, layer=-1):
        """ Inference 1 dataset
        Args:
          ds: dataset
          layer:   [CLS]
        Returns:

        """
        softmax = nn.Softmax(dim = 1)
        dataloader1 = DataLoader(ds, shuffle=False, batch_size=self.args.eval_batch_size)
        pred = []
        feat = []
        label = []
        self.model.eval()
        for batch in dataloader1:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                            'output_hidden_states': True
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits, feats = outputs[0], outputs[1], outputs[2]
                pred_probs = softmax(logits).detach().cpu().numpy()
                pred.append(pred_probs)
                feat.append(feats[layer][:, 0, :].detach().cpu().numpy()) # cls token
                label.append(batch[3].detach().cpu().numpy())
        feat = np.concatenate(feat, axis = 0)
        label = np.concatenate(label, axis = 0)
        pred = np.concatenate(pred, axis = 0)
        return  pred, feat, label

    
    def inference(self, layer = -1):
        ## Inference the embeddings/predictions for unlabeled data
        print("-------------- Evaluating TrainSet...")
        train_pred, train_feat, train_label = self._inf1ds(self.train_dataset) 
        print("train size:", train_pred.shape, train_feat.shape, train_label.shape)
        print("-------------- Evaluating Unlabeled Set")
        unlabeled_pred, unlabeled_feat, unlabeled_label = self._inf1ds(self.unlabeled)
        print("unlabeled size:", unlabeled_pred.shape, unlabeled_feat.shape, unlabeled_label.shape)
        return train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label


    def inference_mc(self, layer = -1):
    # BALD baseline
        print("Evaluating Unlabeled Set icml17bald, monte carlo(mc) sampling")
        #Monte Carlo inference 。
        mc_samples = 8# BatchBALD10，image classification
        # Evaluation of Dpool - MC dropout
        test_losses = []
        logits_list = []
        unlabeled_label = [u[3] for u in self.unlabeled]
        preds = None
        for i in range(mc_samples):
            logits_mc = None
            unlabeled_dataloader = DataLoader(self.unlabeled, shuffle=False, batch_size=self.args.eval_batch_size)
            for batch in unlabeled_dataloader:
                self.model.train() #MC dropout
                batch = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    inputs = {
                                'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2],
                                'labels': batch[3],
                                'output_hidden_states': True
                            }
                    outputs = self.model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]
                    logits= logits.detach().cpu()
                if logits_mc is None:
                    logits_mc = logits
                else:
                    logits_mc = torch.cat((logits_mc, logits), 0)
                # logits_mc.append(logits)

            logits_list.append(logits_mc)
            preds = None

        unlabeled_label = np.array(unlabeled_label)
        return None, None, None,                  logits_list, None, unlabeled_label


    def inference_alps(self, layer = -1):
        """
        Alps baseline   1MiB
        Use vanila PLM to compute MLM loss
        No sample yet
        Only compute vectors
        """
        print("Evaluating Unlabeled Set alps")
        tmp=self.model.to('cpu')
        del self.model
        torch.cuda.empty_cache()

        model_mlm = RobertaForMaskedLM.from_pretrained("roberta-base")
        model_mlm  = model_mlm.to(self.args.device)
        unlabeled_dataloader = DataLoader(self.unlabeled, shuffle=False, batch_size=self.args.eval_batch_size)
        all_scores_or_vectors = None
        for batch in unlabeled_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            scores_or_vectors = batch_scores_or_vectors(batch, self.args, model_mlm, self.args.tokenizer)
            if all_scores_or_vectors is None:
                all_scores_or_vectors = scores_or_vectors.tolist()  #detach().cpu().numpy() #The returned ndarray and the tensor will share their storage, so changes to the tensor will be reflected in the ndarray and vice versa.
            else:
                all_scores_or_vectors +=  scores_or_vectors.tolist()
        # all_scores_or_vectors = torch.tensor(all_scores_or_vectors)
        # train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_logits  # unlabeled_feat  surprisal embedding
        all_scores_or_vectors = np.array(all_scores_or_vectors )
        self.model = tmp.to(self.args.device)
        return None, None, None,  None,                     all_scores_or_vectors, None
                
    


    def evaluate(self, mode):
        # return 1 more  maf1: macro f1 score
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        # logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self.model.eval()
        print("Evaluating", mode)
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        result.update(result)

        logger.info("***** Eval results *****")
  
        return eval_loss, result["acc"], result["maf1"]
    
    