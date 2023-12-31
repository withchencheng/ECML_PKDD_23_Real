import torch
import numpy as np
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import re
import json
import logging
import copy
import random,os
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

PairInputTasks=['qqp']

task2n_labels = {
    "sst-2":2,
    "agnews":4,
    "pubmed":5,
    "snips":7,
    "stov":10,
}


def init_logger(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    logging.basicConfig(filename = f'{args.output_dir}log_{args.al_method}_{args.task}_s{args.seed}',
                        filemode = 'w',
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def get_warm_model_path(args):
#     # before AL， warm upmodel，
#                 warm_model_path = f'{args.output_dir}/model/checkpoint0-seed{args.seed}-{args.model_type}-{args.al_method}'

def print_model_param(args, model):
    n=5
    print(type(model))
    print('seed', args.seed)
    print('Pretrained part:\n', model.roberta.encoder.layer[0].output.dense.weight[:n, :n])
    print('Random init part:\n',model.classifier.dense.weight[:n, :n])
    # path=f'{args.output_dir}_{args.pid}.pth'
    # print('Save classifier to ', path)
    # torch.save(model.classifier, path)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_doc(x, word_freq):
    stop_words = set(stopwords.words('english'))
    clean_docs = []
    most_commons = dict(word_freq.most_common(min(len(word_freq), 50000)))
    for doc_content in x:
        doc_words = []
        cleaned = clean_str(doc_content.strip())
        for word in cleaned.split():
            if word not in stop_words and word_freq[word] >= 5:
                if word in most_commons:
                    doc_words.append(word)
                else:
                    doc_words.append("<UNK>")
        doc_str = ' '.join(doc_words).strip()
        clean_docs.append(doc_str)
    return clean_docs

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label, text_b=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.text_b = text_b


    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, 
                 e1_mask = None, e2_mask = None, keys=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.keys=keys

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Processor(object):
    """Processor for the text data set """
    def __init__(self, args):
        self.args = args
        self.n_labels =  args.n_labels 
        self.relation_labels = [x for x in range(self.n_labels)]
        self.label2id = {x:x for x in range(self.n_labels)}
        self.id2label = {x:x for x in range(self.n_labels)}

    def read_data(self, filename):
        path = filename
        with open(path, 'r') as f:
            data = f 
            for x in data:
                yield json.loads(x)
        # return data

    def _create_examples(self, data, set_type):
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = d["txt"]
            if "txt_b" in d:
                text_b=d["txt_b"]
            else:
                text_b=None
            label = d["lbl"] 
            if i % 5000 == 0:
                logger.info(d)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label, text_b=text_b))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(
            self.read_data(
                os.path.join(self.args.data_dir, file_to_read))
                , mode)


def load_and_cache_examples(args, tokenizer, mode, size = -1):
    processor = Processor(args)
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        )
    )
    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
   
    # Convert to Tensors and build dataset
    if size > 0 and size<len(features):
        print('Error!! No cutting dataset!!')
        exit(7)
        import random 
        random.shuffle(features)
        features = features[:size]
    else:
        size = len(features)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ids = torch.tensor([ _ for _, f in enumerate(features)], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_ids)
    print(f'{mode} {len(dataset)}  {len(features)}')
    return dataset, processor.n_labels, size


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True,
                                ):
    features = []
    for (ex_index, example) in enumerate(examples[:]):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = tokenizer.tokenize(example.text_a)
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        else:
            tokens_b=[]


        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        if len(tokens_a)+len(tokens_b) > max_seq_len - special_tokens_count:
            continue
            # tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]

        if add_sep_token:
            sep_token = tokenizer.sep_token
            tokens = tokens_a + [sep_token] + tokens_b # ，
        else:
            tokens = tokens_a
        token_type_ids = [0] * len(tokens)

        tokens = [tokenizer.cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids  # Roberta token_type_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Done 

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = int(example.label)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        features.append(
            InputFeatures(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label_id=label_id,
                          )
            )

    return features



def compute_entropy( P ):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nansum(-P * np.log(P), axis=1)