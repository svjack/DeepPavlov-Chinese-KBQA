from conf import *
#main_path = "/Volumes/TOSHIBA EXT/temp/kbqa_portable_prj"

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import transformers
import transformers.adapters.composition as ac
from transformers import (
    AdapterConfig,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    MultiLingAdapterArguments,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


import pandas as pd
import pickle as pkl
from copy import deepcopy
import torch
from scipy.special import softmax
from functools import partial, reduce
import json
from io import StringIO
import re

from transformers import list_adapters, AutoModelWithHeads

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()

import os

p0 = os.path.join(main_path, "sel_ner/ner_data_args.pkl")
assert os.path.exists(p0)
with open(p0, "rb") as f:
    t4 = pkl.load(f)

model_args, data_args, training_args, adapter_args = map(deepcopy, t4)

zh_model = AutoModelWithHeads.from_pretrained("bert-base-chinese")

#config_path = "/Users/svjack/temp/ner_trans/adapter_ner_data/test-sel-ner/sel_ner/adapter_config.json"
#adapter_path = "/Users/svjack/temp/ner_trans/adapter_ner_data/test-sel-ner/sel_ner"
#config_path = "sel_ner/adapter_config.json"
#adapter_path = "sel_ner"
config_path = os.path.join(main_path ,"sel_ner/adapter_config.json")
adapter_path = os.path.join(main_path ,"sel_ner")

config = AdapterConfig.load(config_path)
zh_model.load_adapter(adapter_path, config=config)
zh_model.set_active_adapters(['sel_ner'])

def single_sent_pred(input_text, tokenizer, model):
    input_ = tokenizer(input_text)
    input_ids = input_["input_ids"]
    output = model(torch.Tensor([input_ids]).type(torch.LongTensor))
    output_prob = softmax(output.logits.detach().numpy()[0], axis = -1)
    token_list = tokenizer.convert_ids_to_tokens(input_ids)
    assert len(token_list) == len(output_prob)
    return token_list, output_prob

def single_pred_to_df(token_list, output_prob, label_list):
    assert output_prob.shape[0] == len(token_list) and output_prob.shape[1] == len(label_list)
    pred_label_list = pd.Series(output_prob.argmax(axis = -1)).map(
        lambda idx: label_list[idx]
    ).tolist()
    return pd.concat(list(map(pd.Series, [token_list, pred_label_list])), axis = 1)

def token_l_to_nest_l(token_l, prefix = "##"):
    req = []
    #req.append([])
    #### token_l must startswith [CLS]
    assert token_l[0] == "[CLS]"
    for ele in token_l:
        if not ele.startswith(prefix):
            req.append([ele])
        else:
            req[-1].append(ele)
    return req

def list_window_collect(l, w_size = 1, drop_NONE = False):
    assert len(l) >= w_size
    req = []
    for i in range(len(l)):
        l_slice = l[i: i + w_size]
        l_slice += [None] * (w_size - len(l_slice))
        req.append(l_slice)
    if drop_NONE:
        return list(filter(lambda x: None not in x, req))
    return req

def same_pkt_l(l0, l1):
    l0_size_l = list(map(len, l0))
    assert sum(l0_size_l) == len(l1)
    cum_l0_size = np.cumsum(l0_size_l).tolist()
    slice_l = list_window_collect(cum_l0_size, 2, drop_NONE=True)
    slice_l = [[0 ,slice_l[0][0]]] + slice_l
    slice_df = pd.DataFrame(slice_l)
    return (l0, slice_df.apply(lambda s: l1[s[0]:s[1]], axis = 1).tolist())


def cnt_backtrans_slice(token_list, label_list, prefix = "##",
    token_agg_func = lambda x: x[0] if len(x) == 1 else "".join([x[0]] + list(map(lambda y: y[len("##"):], x[1:]))),
    label_agg_func = lambda x: x[0] if len(x) == 1 else pd.Series(x).value_counts().index.tolist()[0]
                       ):
    token_nest_list = token_l_to_nest_l(token_list, prefix=prefix)
    token_nest_list, label_nest_list = same_pkt_l(token_nest_list, label_list)
    token_list_req = list(map(token_agg_func, token_nest_list))
    label_list_req = list(map(label_agg_func, label_nest_list))
    return (token_list_req, label_list_req)

def from_text_to_final(input_text, tokenizer, model, label_list):
    token_list, output_prob = single_sent_pred(input_text, tokenizer, model)
    token_pred_df = single_pred_to_df(token_list, output_prob, label_list)
    token_list_, label_list_ = token_pred_df[0].tolist(), token_pred_df[1].tolist()
    token_pred_df_reduce = pd.DataFrame(list(zip(*cnt_backtrans_slice(token_list_, label_list_))))
    return token_pred_df_reduce


tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
)

label_list = ['O-TAG', 'E-TAG', 'T-TAG']

if __name__ == "__main__":
    from_text_to_final("宁波在哪个省？",
    tokenizer,
    zh_model,
    label_list
                  )

    from_text_to_final("美国的通货是什么？",
    tokenizer,
    zh_model,
    label_list
                  )
