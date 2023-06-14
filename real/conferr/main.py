import argparse
from datetime import datetime
import os
from real.conferr.utils import set_seed, load_and_cache_examples, init_logger, load_tokenizer, task2n_labels
from real.conferr.trainer import Trainer
from real.conferr.opts import *

def model_dict(model_type):
    if model_type == 'roberta-base':
        return 'roberta-base'
    elif model_type == 'bert-base':
        return 'bert-base-uncased'


def main(args):
    init_logger(args)
    set_seed(args.seed)
    tokenizer = load_tokenizer(args)
    args.tokenizer = tokenizer
    
    dev_dataset, num_labels, dev_size = load_and_cache_examples(args, tokenizer, mode="dev", size = args.dev_labels)
    test_dataset, num_labels, test_size = load_and_cache_examples(args, tokenizer, mode="test")
    unlabeled_dataset, num_labels, unlabeled_size = load_and_cache_examples(args, tokenizer, mode = 'train', size = 100000) # all in train.json

    print('number of labels:', num_labels)
    print('dev_size:', dev_size)
    print('test_size:', test_size)
    print('total unlabel_size:', unlabeled_size)

    # dseltrain={ # dict of trainer selector
    #     0: Trainer,
    #     1: TrainerTriplet
    # }
    trainer = Trainer(args, train_dataset=None, dev_dataset=dev_dataset,test_dataset=test_dataset, \
            unlabeled = unlabeled_dataset, \
            num_labels = num_labels
            )
    
    trainer.move_model_device()

    home_dir= os.path.expanduser( '~/' ) 
    warm_model_path = f'{home_dir}data/AL/experiment/warm_model/{args.task}_wsz{args.warm_size}_s{args.seed}.pth'

    for i in range(args.rounds+1): 
        if i == 0:
            if os.path.exists(warm_model_path):
                trainer.load_model(warm_model_path)
                loss_test, acc_test, maf1_test = trainer.evaluate('test')
                print(f'Round {i}, Test: Loss: {loss_test}, Acc: {acc_test}, Macro-f1: {maf1_test}')
                trainer.tb_writer.add_scalar(f"Test_acc_{args.al_method}_seed{args.seed}", acc_test, i)
            else:
                trainer.train(round=i)
                # trainer.save_model(warm_model_path)
        else:
            #---------sample
            args.num_train_epochs = 4 
            trainer.sample(n_sample = args.sample_labels, n_unlabeled = -1, round = i)
            trainer.train(round=i)
            

    trainer.close_clean()

    

if __name__ == '__main__':
    pid = os.getpid()
    dt_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f'Pid {pid}, Started at {dt_string}')

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str, help="")

    parser.add_argument("--seed", default=0, type=int, help="which seed to use")
    parser.add_argument("--warm_size", default=100, type=int, help="Initial warm training size.") 
    parser.add_argument("--task", default="agnews", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="../datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
    parser.add_argument("--tsb_dir", default="./eval", type=str, help="TSB script, result directory")
    parser.add_argument("--train_file", default="", type=str, help="Train file")
    parser.add_argument("--dev_file", default="", type=str, help="dev file")
    parser.add_argument("--test_file", default="", type=str, help="Test file")
    parser.add_argument("--unlabel_file", default="", type=str, help="Test file")
    parser.add_argument("--sample_labels", default=100, type=int, help="number of labels for sampling in AL")
    parser.add_argument("--dev_labels", default=1000, type=int, help="number of labels for dev set")
    parser.add_argument("--neval1epoch", default=4, type=int, help="eval dev times per epoch")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--unctth", default=None, type=float, help="Uncertainty threshold proportion.uncertainty top 100")
    parser.add_argument("--ncentroids", default=25, type=int, help=" uncterr1.1。#clusters")
    # #

    parser.add_argument('--rounds', type=int, default=10, help="Active Learning Rounds.")
    parser.add_argument('--logging_steps', type=int, default=10, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument("--model_type", default="bert-base-uncased", type=str)
    parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence")
    parser.add_argument("--save_sample", action="store_true", help="Add [SEP] token at the end of the sentence")

    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=1500, type=int, help="Training steps for initialization.")# no use
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=512, type=int, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="The maximum total input sequence length after tokenization.")


    parser.add_argument("--al_method", default='random', type=str, help="The initial learning rate for Adam.")

    # triplet_opts(parser)


    args = parser.parse_args()
    args.model_name_or_path = model_dict(args.model_type)
    args.pid = pid
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.n_labels = task2n_labels[args.task]
    
    print('ARGS:\n',args)
    main(args)