"""
Code from https://github.com/forest-snow/alps
"""
import argparse
import os

import torch
import random
import numpy as np
from transformers import glue_processors as processors
from transformers import (
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)
# from src.data import processors, output_modes


MODELS = {
    "lm":AutoModelWithLMHead,
    "sc":AutoModelForSequenceClassification
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_model(args):
    args.model_type = args.model_type.lower()
    if args.head=="sc":
        processor = processors[args.task_name]()
        # args.output_mode = output_modes[args.task_name]
        label_list = processor.get_labels()
        num_labels = len(label_list)
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer_class = AutoTokenizer
    model_class = MODELS[args.head]
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)
    return model, tokenizer, model_class, tokenizer_class


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument("--base_model",
        type=str,
        required=True,
        help="The base model (for active learning experiments) name or path for loading cached data"
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument("--eval_pretrained", action="store_true", help="Only evaluate pretrained model.")
    parser.add_argument(
        "--do_lower_case", default=True, type=bool, help="Set this flag if you are using an uncased model.",
    )


    # Additional arguments for active learning
    parser.add_argument("--sampling", type=str, help="Acquisition function for active learning.")
    parser.add_argument("--query_size", default=0, type=int, help="Size of acquisition")
    parser.add_argument(
        "--masked", action="store_true", help="Set this flag if want input masked for active learning"
    )
    parser.add_argument(
        "--mlm", default=True, type=bool, help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--server",
        type=str,
        # default='ford',
        default='jade',
        help=" which server",
    )
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.server == 'ford':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        args.n_gpu = 1
        device = torch.device('cuda:0')
    else:
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            args.n_gpu = 1
    setattr(args, 'device', device)
    print('device {}'.format(device))
    return args
