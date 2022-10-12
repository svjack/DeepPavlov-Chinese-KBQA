#main_path = "/Users/svjack/temp/gradio_prj"

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
#from django.db.transaction import commit_on_success
from django.db import transaction

'''
from .models import Embedding
from .models import Translation
from .models import Valid
'''
import json
import os

#main_path = "/Volumes/TOSHIBA EXT/temp/kbqa_portable_prj"
#main_path = "/User/kbqa_portable_prj"
main_path = "/temp/kbqa_portable_prj"

import json
import os
from functools import lru_cache, partial, reduce

import numpy as np
import pandas as pd
import sqlite_utils
from rdflib.graph import Graph
from rdflib_hdt import HDTStore
from timer import timer
from tqdm import tqdm

os.environ["DP_SKIP_NLTK_DOWNLOAD"] = "True"

import inspect
import json
import logging
import os
import re
import sys
from collections import defaultdict
from functools import reduce
from itertools import permutations, product

import numpy as np
import pandas as pd
from deeppavlov import build_model, configs
from deeppavlov.core.commands.infer import *
from deeppavlov.core.commands.utils import *
from deeppavlov.core.common.file import *
from deeppavlov.models.kbqa.wiki_parser import *
from rapidfuzz import fuzz
from scipy.special import softmax


logging.disable(sys.maxsize)

import csv
import gzip
import inspect
import logging
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict, namedtuple
from copy import deepcopy
from datetime import datetime
from functools import partial, reduce

import editdistance
import networkx as nx
import numpy as np
import pandas as pd
import sqlite_utils
import synonyms
import torch.nn as nn
from deeppavlov import build_model, configs
from deeppavlov.core.commands.infer import *
from deeppavlov.core.common.file import *
from deeppavlov.models.kbqa.query_generator import *
from deeppavlov.models.kbqa.query_generator_base import *
from deeppavlov.models.kbqa.wiki_parser import *
from pandas.io.common import _stringify_path
from scipy.special import softmax
from sentence_transformers import InputExample, LoggingHandler, util
from sentence_transformers.util import pytorch_cos_sim
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import \
    CECorrelationEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

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

from dataclasses import dataclass, field

import jieba
from hashlib import sha512

pd.set_option('max_colwidth', 60)
pd.set_option("max_columns", 20)

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

import sys
sys.path.insert(0, main_path)

from ner_model import *
from tmp_classifier import *
from ranker import *

#### or load specific version
#p0 = os.path.join(main_path, "LaBSE_local")
p0 = os.path.join("/temp/model", "LaBSE_local")
sim_model = SentenceTransformer(p0)
#sim_model = SentenceTransformer('LaBSE')
sim_model.pool = None

'''
p1 = os.path.join(main_path, "kbqa-explore/multi_lang_kb_dict.db")
#assert os.path.exists("kbqa-explore/multi_lang_kb_dict.db")
assert os.path.exists(p1)
#wiki_entity_db = sqlite_utils.Database("kbqa-explore/multi_lang_kb_dict.db")
wiki_entity_db = sqlite_utils.Database(p1)
assert "en_zh_so_search" in wiki_entity_db.table_names()
'''
#wiki_entity_db = pd.read_csv(os.path.join(main_path, "kbqa-explore/db_dump.csv"), header = None)
wiki_entity_db = pd.read_csv(os.path.join("/temp", "kbqa-explore/db_dump.csv"), header = None)
#assert "en_zh_so_search" in wiki_entity_db.table_names()
wiki_entity_db = wiki_entity_db.iloc[:, 1:]
assert wiki_entity_db.shape[1] == 3
wiki_entity_db.columns = ["s", "o", "lang"]

#hdt_path = "/Users/svjack/.deeppavlov/downloads/wikidata/wikidata.hdt"
#hdt_path = os.path.join(main_path, "kbqa-explore/wikidata.hdt")
hdt_path = os.path.join("/temp", "kbqa-explore/wikidata.hdt")
assert os.path.exists(hdt_path)
wiki_parser = WikiParser(
    wiki_filename = hdt_path,
    lang = "",
)

#p2 = os.path.join(main_path, "kbqa-explore")
p2 = os.path.join("/temp", "kbqa-explore")
assert os.path.exists(p2)
sys.path.insert(0, p2)

#p3 = os.path.join(main_path, "kbqa-explore/linker_entities.pkl")
p3 = os.path.join("/temp", "kbqa-explore/linker_entities.pkl")
assert os.path.exists(p3)
#zh_linker_entities = load_pickle("kbqa-explore/linker_entities.pkl")
zh_linker_entities = load_pickle(p3)
zh_linker_entities.num_entities_to_return = 5

config = parse_config(configs.kbqa.kbqa_cq)

#### query template info
#sparql_queries_path = pd.DataFrame(config["chainer"]["pipe"])["sparql_queries_filename"].dropna().iloc[0]
sparql_queries_path = os.path.join(main_path, "sparql_queries.json")
assert os.path.join(sparql_queries_path)
sparql_queries_df = pd.read_json(sparql_queries_path).T

'''
def search_entity_rep_by_lang_filter_in_db(entityid, wiki_entity_db, lang = "en"):
    id = entityid
    g = wiki_entity_db.query("select * from en_zh_so_search where s = '{}' and lang = '{}'".format(id, lang))
    l = list(g)
    if not l:
        return []
    df = pd.DataFrame(l)
    l = df["o"].drop_duplicates().tolist()
    return l
'''
def search_entity_rep_by_lang_filter_in_db(entityid, wiki_entity_db, lang = "en"):
    id = entityid
    '''
    g = wiki_entity_db.query("select * from en_zh_so_search where s = '{}' and lang = '{}'".format(id, lang))
    l = list(g)
    '''
    df = wiki_entity_db[
        wiki_entity_db["s"] == entityid
    ]
    if not df.size:
        return []
    df = df[
        df["lang"] == lang
    ]
    if not df.size:
        return []
    '''
    if not l:
        return []
    '''
    #df = pd.DataFrame(l)
    l = df["o"].drop_duplicates().tolist()
    return l

class Zh_Rel_Ranker(object):
    def __init__(self, rfr_cls, pid_text_b_dict):
        assert hasattr(rfr_cls, "produce_rank_df")
        assert hasattr(rfr_cls, "sim_model")
        rfr_cls.sim_model = SentenceTransformer("LaBSE")
        assert type(pid_text_b_dict) == type({})
        self.rfr_cls = rfr_cls
        self.pid_text_b_dict = pid_text_b_dict
        self.text_b_pid_dict = dict(map(lambda t2: (t2[1], t2[0])
                                   ,self.pid_text_b_dict.items()))

    def rank_rels(self, question, ex_rels, reverse_to_id = True):
        assert type(question) == type("")
        assert type(ex_rels) == type([])
        ex_rels = list(filter(lambda x: x in self.pid_text_b_dict.keys(), ex_rels))
        ex_rels_text_list = []
        for x in ex_rels:
            text_b = self.pid_text_b_dict[x]
            ex_rels_text_list.append(text_b)
        prob_cate_df = self.rfr_cls.produce_rank_df(question, ex_rels_text_list)
        req = prob_cate_df[["cate", "prob"]].values.tolist()
        if reverse_to_id:
            req = list(map(
                lambda t2: (self.text_b_pid_dict[t2[0]] ,t2[1])
                , req))
        return req

br_cls = RFR(b_clf,
    all_cate_list=list(pid_zh_b_dict.values()),
   sim_model=sim_model
   )
b_rel_ranker = Zh_Rel_Ranker(br_cls, pid_zh_b_dict)

def query_parser_bu(question ,query_info,
                     entities_and_types_select,
                   entity_ids,
                     type_ids,
                    entities_to_leave = 10,
                     max_comb_num = 10000,
                     return_all_possible_answers = False,
                     rel_ranker = b_rel_ranker
                   ):
    question_tokens = jieba.lcut(question)
    query = query_info["query_template"].lower()

    rels_for_search = query_info["rank_rels"]
    rel_types = query_info["rel_types"]
    query_seq_num = query_info["query_sequence"]
    return_if_found = query_info["return_if_found"]
    define_sorting_order = query_info["define_sorting_order"]
    property_types = query_info["property_types"]

    query_triplets = re.findall("{[ ]?(.*?)[ ]?}", query)[0].split(' . ')
    query_triplets = [triplet.split(' ')[:3] for triplet in query_triplets]
    query_sequence_dict = {num: triplet for num, triplet in zip(query_seq_num, query_triplets)}
    query_sequence = []
    for i in range(1, max(query_seq_num) + 1):
        query_sequence.append(query_sequence_dict[i])
    triplet_info_list = [("forw" if triplet[2].startswith('?') else "backw", search_source, rel_type)
                             for search_source, triplet, rel_type in zip(rels_for_search, query_triplets, rel_types) if
                             search_source != "do_not_rank"]
    entity_ids = [entity[:entities_to_leave] for entity in entity_ids]

    rels = [find_top_rels_bu(question, entity_ids, triplet_info, wiki_parser, rel_ranker)
                    for triplet_info in triplet_info_list]
    log.debug(f"(query_parser)rels: {rels}")
    rels_from_query = [triplet[1] for triplet in query_triplets if triplet[1].startswith('?')]
    answer_ent = re.findall("select [\(]?([\S]+) ", query)
    order_info_nt = namedtuple("order_info", ["variable", "sorting_order"])
    order_variable = re.findall("order by (asc|desc)\((.*)\)", query)
    if order_variable:
        if define_sorting_order:
            answers_sorting_order = order_of_answers_sorting(question)
        else:
            answers_sorting_order = order_variable[0][0]
        order_info = order_info_nt(order_variable[0][1], answers_sorting_order)
    else:
        order_info = order_info_nt(None, None)
    print(f"question, order_info: {question}, {order_info}")
    filter_from_query = re.findall("contains\((\?\w), (.+?)\)", query)
    print(f"(query_parser)filter_from_query: {filter_from_query}")
    year = extract_year(question_tokens, question)
    number = extract_number(question_tokens, question)
    print(f"year {year}, number {number}")
    if year:
        filter_info = [(elem[0], elem[1].replace("n", year)) for elem in filter_from_query]
    elif number:
        filter_info = [(elem[0], elem[1].replace("n", number)) for elem in filter_from_query]
    else:
        filter_info = [elem for elem in filter_from_query if elem[1] != "n"]
    for unk_prop, prop_type in property_types.items():
        filter_info.append((unk_prop, prop_type))
    print(f"(query_parser)filter_from_query: {filter_from_query}")
    rel_combs = make_combs(rels, permut=False)
    import datetime
    start_time = datetime.datetime.now()
    entity_positions, type_positions = [elem.split('_') for elem in entities_and_types_select.split(' ')]
    print(f"entity_positions {entity_positions}, type_positions {type_positions}")
    selected_entity_ids = [entity_ids[int(pos) - 1] for pos in entity_positions if int(pos) > 0]
    selected_type_ids = [type_ids[int(pos) - 1] for pos in type_positions if int(pos) > 0]
    entity_combs = make_combs(selected_entity_ids, permut=True)
    type_combs = make_combs(selected_type_ids, permut=False)
    print(f"(query_parser)entity_combs: {entity_combs[:3]}, type_combs: {type_combs[:3]},"
              f" rel_combs: {rel_combs[:3]}")
    queries_list = []
    parser_info_list = []
    confidences_list = []
    all_combs_list = list(itertools.product(entity_combs, type_combs, rel_combs))

    for comb_num, combs in enumerate(all_combs_list):
        confidence = np.prod([score for rel, score in combs[2][:-1]])
        confidences_list.append(confidence)
        query_hdt_seq = [
            fill_query(query_hdt_elem, combs[0], combs[1], combs[2]) for query_hdt_elem in query_sequence]
        if comb_num == 0:
            print(f"\n__________________________\nfilled query: {query_hdt_seq}\n__________________________\n")
        queries_list.append((rels_from_query + answer_ent, query_hdt_seq, filter_info, order_info, return_if_found))
        parser_info_list.append("query_execute")
        ##if comb_num == self.max_comb_num:
        if comb_num == max_comb_num:
            break
    candidate_outputs = []
    #candidate_outputs_list = self.wiki_parser(parser_info_list, queries_list)
    candidate_outputs_list = wiki_parser(parser_info_list, queries_list)

    if isinstance(candidate_outputs_list, list) and candidate_outputs_list:
        outputs_len = len(candidate_outputs_list)
        all_combs_list = all_combs_list[:outputs_len]
        confidences_list = confidences_list[:outputs_len]
        for combs, confidence, candidate_output in zip(all_combs_list, confidences_list, candidate_outputs_list):
            candidate_outputs += [[combs[0]] + [rel for rel, score in combs[2][:-1]] + output + [confidence]
                                  for output in candidate_output]
        #if self.return_all_possible_answers:
        if return_all_possible_answers:
            candidate_outputs_dict = defaultdict(list)
            for candidate_output in candidate_outputs:
                candidate_outputs_dict[(tuple(candidate_output[0]),
                                        tuple(candidate_output[1:-2]))].append(candidate_output[-2:])
            candidate_outputs = []
            for (candidate_entity_comb, candidate_rel_comb), candidate_output in candidate_outputs_dict.items():
                candidate_outputs.append(list(candidate_rel_comb) +
                                         [tuple([ans for ans, conf in candidate_output]), candidate_output[0][1]])
        else:
            candidate_outputs = [output[1:] for output in candidate_outputs]
    print(f"(query_parser)loop time: {datetime.datetime.now() - start_time}")
    print(f"(query_parser)final outputs: {candidate_outputs[:3]}")
    return candidate_outputs

def find_top_rels_bu(question: str, entity_ids: List[List[str]],
                     triplet_info: Tuple, wiki_parser, rel_ranker,
                     rels_to_leave = 10
                     , entities_to_leave = 5, source = "wiki",
                    ):
    assert source == "wiki"
    ex_rels = []
    direction, source, rel_type = triplet_info
    if source == "wiki":
        print(triplet_info)
        print("entity_ids :")
        print(entity_ids)
        print("-" * 100)

        queries_list = list({(entity, direction, rel_type) for entity_id in entity_ids
                             for entity in entity_id[:entities_to_leave]})
        print("queries_list: {}".format(queries_list))

        parser_info_list = ["find_rels" for i in range(len(queries_list))]
        ex_rels = wiki_parser(parser_info_list, queries_list)
        ex_rels = list(set(ex_rels))
        ex_rels = [rel.split('/')[-1] for rel in ex_rels]

    #return question, ex_rels
    print("ex_rels :")
    print(ex_rels)

    rels_with_scores = rel_ranker.rank_rels(question, ex_rels)
    rels_with_scores = rels_with_scores[:rels_to_leave]
    return rels_with_scores

def t3_statement_df(query):
    assert type(query) == type([])
    assert len(query) == 3
    #print("before call search: ______")
    query_statement_df = search_triples_with_parse(
wiki_parser.document, query
).applymap(str).applymap(
    lambda x: search_entity_rep_by_lang_filter_in_db(x.split("/")[-1], wiki_entity_db, "zh") \
    if type(x) == type("") and x.startswith("http://www.wikidata.org/entity/Q") else x
)
    query_statement_df["p"] = query_statement_df["p"].map(
        lambda x: pid_zh_b_dict.get(x.split("/")[-1], "").split(" ") if x.split("/")[-1].startswith("P") else []
    )
    query_statement_df["s"] = query_statement_df["s"].map(
        lambda x: x if type(x) == type([]) else [str(x)]
    )
    query_statement_df["o"] = query_statement_df["o"].map(
        lambda x: x if type(x) == type([]) else [str(x)]
    )

    return query_statement_df

def fix_o(o, rm_char = ["\\"]):
    if not  o.startswith('"'):
        return o
    #print(o)
    assert o.startswith('"')
    num = []
    for i in range(len(o)):
        c = o[i]
        if c == '"':
            num.append(i)
    assert len(num) >= 2
    rm_num = num[1:-1]
    return "".join(
        map(lambda ii: o[ii], filter(lambda i: i not in rm_num and o[i] not in rm_char, range(len(o))))
    )

def py_dumpNtriple(
    subject, predicate, object_
):
    #### java rdfhdt dumpNtriple python format
    out =[]
    s0 = subject[0]
    if s0=='_' or s0 =='<':
        out.append(subject);
    else:
        out.append('<')
        out.append(subject)
        out.append('>')

    p0 = predicate[0]
    if p0=='<':
        out.append(' ')
        out.append(predicate)
        out.append(' ');
    else:
        out.append(" <")
        out.append(predicate)
        out.append("> ")

    o0 = object_[0]
    if o0=='"':
        #out.append(object_)
        ####
        #UnicodeEscape.escapeString(object.toString(), out);
        #out.append(json.dumps([object_])[1:-1])
        out.append(object_)
        out.append(" .\n");
    elif o0=='_' or o0=='<':
        out.append(object_)
        out.append(" .\n")
    else:
        out.append('<')
        out.append(object_)
        out.append("> .\n")
    return "".join(out)

def one_part_g_producer(one_part_string,
                       format_ = "nt"
                       ):
    from uuid import uuid1
    from rdflib import Graph

    tmp_f_name = "{}.{}".format(uuid1(), format_)
    with open(tmp_f_name, "w") as f:
        f.write(one_part_string)
    g = Graph()
    g.parse(tmp_f_name, format=format_)

    os.remove(tmp_f_name)
    return g

def drop_duplicates_by_col(df, on_col = "aug_sparql_query"):
    assert hasattr(df, "size")
    assert on_col in df.columns.tolist()
    req = []
    set_ = set([])
    for i, r in df.iterrows():
        if r[on_col] not in set_:
            set_.add(r[on_col])
            req.append(r)
    return pd.DataFrame(req)

def drop_duplicates_of_every_df(df):
    if not df.size:
        return df
    ori_columns = df.columns.tolist()
    df["hash"] = df.apply(lambda s: sha512(str(s.to_dict()).encode()).hexdigest(), axis = 1)
    req = []
    k_set = set([])
    for i, r in df.iterrows():
        if r["hash"] not in k_set:
            req.append(r.to_dict())
        k_set.add(r["hash"])
    return pd.DataFrame(req)[ori_columns]

def search_triples_with_parse(source ,query, return_df = True, skip_some_o = True, max_times = int(1e3)):
    assert hasattr(source, "search_triples")
    #print("before search_triples: ______")
    iter_, num = source.search_triples(*query)
    req = []
    for s, p, o in iter_:
        o = fix_o(o)
        if skip_some_o:
            if "\n" in o:
                continue
        nt_str = py_dumpNtriple(s, p, o)
        req.append(nt_str)
        if len(req) >= max_times:
            break
    #print("before one_part_g_producer: ______")
    g = one_part_g_producer("".join(req))
    #print("before return in search_triples_with_parse: ______")
    if return_df:
        return pd.DataFrame(g.__iter__(), columns = ["s", "p", "o"])
    return g

def perm_top_sort(en_sent ,zh_perm_list, model, return_score = False):
    assert len(zh_perm_list) >= 1
    if len(zh_perm_list) == 1:
        return zh_perm_list[0]
    #### zh_perm_list length too big problem

    embedding = model.encode([en_sent] + zh_perm_list)

    sim_m = pytorch_cos_sim(embedding, embedding)
    sim_a = sim_m[0]
    if return_score:
        return sim_a.numpy()
    #### same top val 1
    max_index = np.argsort(sim_a.numpy()[1:])[-1]
    return zh_perm_list[max_index]

def syn_sim_on_list(sent, l):
    assert type(l) == type([])
    sim_df = pd.DataFrame(pd.Series(l).drop_duplicates().map(
        lambda x: (x,
        (synonyms.compare(sent, " ".join(re.findall(u"[\u4e00-\u9fa5]+", x)))\
         + (fuzz.ratio(sent, x) / 100.0)) / 2.0
        )
    ).values.tolist()
    )
    sim_df.columns = ["zh_info", "score"]
    sim_df = sim_df.sort_values(by = "score", ascending = False)
    return sim_df

def t3_statement_ranking(
    question,
    entity_list = ["http://www.wikidata.org/entity/Q42780"],
    property_list = ["http://www.wikidata.org/prop/direct/P131",
                    "http://www.wikidata.org/prop/direct/P150"
    ],
    generate_t3_func = lambda el, pl: pd.Series(list(product(el, pl))).map(
        lambda ep: [(ep[0], ep[1], ""), ("", ep[1], ep[0])]
    ).explode().dropna().drop_duplicates().tolist(),
    clf = b_clf,
    show_query = False,
    use_ranker = False,
):
    query_list = list(map(list ,generate_t3_func(entity_list, property_list)))
    #print(query_list)

    #df_list = list(map(t3_statement_df, query_list))
    df_list = []
    #print("before tqdm query_list: ______")
    for ele in tqdm(query_list):
        if show_query:
            print(ele)
        df_list.append(t3_statement_df(ele))

    #return df_list

    assert len(query_list) == len(df_list)
    query_list_ = []
    df_list_ = []
    for i in range(len(query_list)):
        df = df_list[i]
        if hasattr(df, "size") and df.size > 0:
            query_list_.append(query_list[i])
            df_list_.append(df_list[i])
    assert len(query_list_) == len(df_list_)
    if len(query_list_) == 0:
        return None
    query_list = query_list_
    df_list = df_list_

    #print(len(df_list))
    #print("-" * 100)

    df_list = list(map(
        lambda df: df.applymap(
    lambda x: sorted(x, key = lambda y: fuzz.ratio(y, question), reverse = True)[0] if x else np.nan
    ).dropna()
    , df_list))

    query_list_ = []
    df_list_ = []
    for i in range(len(query_list)):
        df = df_list[i]
        if hasattr(df, "size") and df.size > 0:
            query_list_.append(query_list[i])
            df_list_.append(df_list[i])
    assert len(query_list_) == len(df_list_)
    if len(query_list_) == 0:
        return None
    query_list = query_list_
    df_list = df_list_

    #print(len(df_list))
    #print("-" * 100)

    req = []
    for i in range(len(query_list)):
        ele = df_list[i].copy()
        ele["cate"] = [tuple(query_list[i])] * len(ele)
        req.append(ele)

    abcd_df = pd.concat(req, axis = 0)
    abcd_df["key"] = abcd_df[["s", "p", "o"]].apply(lambda x: "".join(x.tolist()), axis = 1)

    ###return abcd_df

    if use_ranker:
        br_cls_s = RFR(b_clf,
            all_cate_list=abcd_df["key"].tolist(),sim_model=sim_model)
        rank_df = br_cls_s.produce_rank_df(question, br_cls_s.all_cate_list)
    else:
        #print("before syn_sim rank :______")

        rank_df = syn_sim_on_list(question, abcd_df["key"].tolist())
        rank_df = rank_df.rename(
            columns = {
                "score": "prob",
                "zh_info": "cate"
            }
        )

        br_cls_s = RFR(b_clf,
            all_cate_list=abcd_df["key"].tolist(),sim_model=sim_model)
        rank_df_ori = br_cls_s.produce_rank_df(question, br_cls_s.all_cate_list)

        rank_df = rank_df.reset_index().iloc[:, 1:]
        rank_df_ori = rank_df_ori.reset_index().iloc[:, 1:]

        '''
        print("rank_df :")
        print(rank_df)
        print("rank_df_ori :")
        print(rank_df_ori)
        print("merge :")
        print(pd.merge(rank_df, rank_df_ori, on = "cate"))
        print("-" * 100)
        '''

        merge_df = pd.merge(rank_df, rank_df_ori, on = "cate")
        cate_list = merge_df["cate"].tolist()
        prob_list = merge_df[["prob_x", "prob_y"]].max(axis = 1).tolist()
        rank_df = pd.concat([pd.Series(cate_list), pd.Series(prob_list)], axis = 1)
        rank_df.columns = ["cate", "prob"]

    abcd_df = pd.merge(rank_df[["cate", "prob"]].rename(
        columns = {
            "cate": "key"
        }
    ), abcd_df, on = "key").sort_values(by = "prob", ascending = False)

    abcd_uni_df = drop_duplicates_by_col(abcd_df, "cate")
    return abcd_uni_df

def choose_tmp_by_ranking(question,
    entity_list,
    tmp_conclusion_dict,
    tmp_generate_t3_func_dict,
    aug_func = max,
):
    assert type(question) == type("")
    assert type(entity_list) == type([])
    assert type(tmp_conclusion_dict) == type(dict())
    assert type(tmp_generate_t3_func_dict) == type(dict())
    assert len(tmp_conclusion_dict) == len(tmp_generate_t3_func_dict)
    assert set(tmp_conclusion_dict.keys()) == set(tmp_generate_t3_func_dict.keys())
    req = {}
    req_df_dict = {}
    for k in tmp_conclusion_dict.keys():
        till_list = tmp_conclusion_dict[k]
        assert type(till_list) == type([])
        if till_list:
            #print(k ,till_list)
            assert len(till_list[0]) == 3
        if not till_list:
            continue
        property_list = pd.Series(till_list).map(
            lambda t3: list(map(
                lambda x: x.format(t3[0]),
                ["http://www.wikidata.org/prop/direct/{}",
                "http://www.wikidata.org/prop/statement/{}",
                "http://www.wikidata.org/prop/{}",]
            ))
        ).explode().dropna().drop_duplicates().tolist()

        generate_t3_func = tmp_generate_t3_func_dict[k]
        assert callable(generate_t3_func)

        #print("before t3_statement_ranking: ____")
        ranking_df = t3_statement_ranking(
            question = question,
            entity_list = entity_list,
            property_list = property_list,
            generate_t3_func = generate_t3_func
        )

        if not hasattr(ranking_df, "size") or ranking_df.size == 0:
            continue
        print("k ", k)
        print(ranking_df.head(3))

        ###assert ranking_df.shape[1] == 3
        assert "cate" in ranking_df.columns.tolist()
        assert "prob" in ranking_df.columns.tolist()
        score = aug_func(ranking_df["prob"].values.tolist())
        req[k] = score
        req_df_dict[k] = ranking_df
    for kk in tmp_conclusion_dict.keys():
        if kk not in req:
            req[kk] = -1
        if kk not in req_df_dict:
            req_df_dict[kk] = None
    ###return req
    best_tmp_cate = sorted(req.items(), key = lambda t2: t2[1], reverse = True)[0][0]
    assert best_tmp_cate in tmp_conclusion_dict
    ###best_ranking_df = req_df_dict[kk]
    best_ranking_df = req_df_dict[best_tmp_cate]

    a, b, c = best_tmp_cate, tmp_conclusion_dict[best_tmp_cate], best_ranking_df

    b_df = pd.DataFrame(b)

    if False:
        pass
    else:
        b_df[1] = b_df[1].map(
    lambda x:
    search_entity_rep_by_lang_filter_in_db(x.split("/")[-1], wiki_entity_db, "zh")\
    if type(x) == type("") and x.startswith("http://www.wikidata.org/entity/Q") else str(x)
).map(
    lambda x: x if type(x) != type("") else (re.findall('"(.+)"', x)[0] if len(x.split('"')) >= 3 else x)
)

    b_df.columns = ["pid", "entity", "score"]
    b_df = b_df.sort_values(by = "score", ascending = False)
    if c is None:
        c_dict = {"": 1.0}
    else:
        #c_dict = dict(c[["o", "prob"]].values.tolist())
        c_dict = dict(pd.DataFrame(pd.DataFrame(c.apply(
    lambda x:
    list(
    map(lambda i: (np.nan if x["cate"][i] else x[["s", "p", "o"]].tolist()[i], x["prob"]) ,range(len(x["cate"])))
    )
    , axis = 1).values.tolist()).values.reshape([-1]).tolist()).dropna().values.tolist())
        c_dict_ = {}
        for k, v in c_dict.items():
            c_dict_[str(k)] = v
        c_dict = c_dict_

    #return b_df, c_dict

    b_df_rewighted = pd.DataFrame(b_df.apply(
    lambda x:
        (x["pid"], x["entity"], x["score"] * \
        (max(map(lambda y:
    c_dict.get(y, min(c_dict.values())), x["entity"])) if x["entity"] and type(x["entity"]) == type([]) else \
    c_dict.get(x["entity"], min(c_dict.values())) if type(x["entity"]) == type("") else min(c_dict.values())
    )
    )
    , axis = 1
    ).values.tolist())
    b_df_rewighted.columns = b_df.columns.tolist()

    b_df_rewighted = b_df_rewighted.rename(
        columns = {"score": "multi_score"}
    )
    b_df_rewighted["score"] = b_df["score"].tolist()
    b_df_rewighted["rank_score"] = b_df_rewighted["multi_score"] / b_df_rewighted["score"]
    b_df_rewighted["max_score"] = b_df_rewighted[["score", "rank_score"]].apply(
        lambda x: x.max(), axis = 1
    )
    b_df_rewighted["sum_score"] = b_df_rewighted[["score", "rank_score"]].apply(
        lambda x: x.sum(), axis = 1
    )

    b_df_rewighted = b_df_rewighted.sort_values(by = "sum_score", ascending = False)

    return a, b_df_rewighted, best_ranking_df

def till_process_func(till_list):
    assert type(till_list) == type([])
    if not till_list:
        return till_list
    ele_length = list(map(len, till_list))
    assert len(set(ele_length)) == 1
    ele_length = ele_length[0]
    assert ele_length in [3, 4]
    if ele_length == 3:
        return till_list
    def filter_row(row_list):
        assert type(row_list) == type([])
        left = row_list[0]
        right = row_list[-1]
        mid = row_list[1:-1]
        assert mid
        if len(mid) == 1:
            return [left, mid[0], right]
        mid = list(filter(lambda x: not x.startswith("http://www.wikidata.org/prop/"), mid))
        assert mid
        return [left, mid[0], right]
    return list(map(filter_row, till_list))

### fix eng with " "
### used when ner_model input with some eng-string fillwith " "
def fill_str(sent ,str_):
    is_en = False
    if re.findall("[a-zA-Z0-9 ]+", str_) and re.findall("[a-zA-Z0-9 ]+", str_)[0] == str_:
        is_en = True
    if not is_en:
        return str_
    find_part = re.findall("([{} ]+)".format(str_), text)
    assert find_part
    find_part = sorted(filter(lambda x: x.replace(" ", "") == str_.replace(" ", "") ,find_part), key = len, reverse = True)[0]
    assert find_part in sent
    return find_part

def for_loop_detect(s, invalid_tag = "O-TAG", sp_token = "123454321"):
    assert type(s) == type(pd.Series())
    char_list = s.iloc[0]
    tag_list = s.iloc[1]
    assert len(char_list) == len(tag_list)
    req = defaultdict(list)
    pre_tag = ""
    for idx, tag in enumerate(tag_list):
        if tag == invalid_tag or tag != pre_tag:
            for k in req.keys():
                if req[k][-1] != invalid_tag:
                    req[k].append(sp_token)
            if tag != pre_tag and tag != invalid_tag:
                char = char_list[idx]
                req[tag].append(char)
        elif tag != invalid_tag:
            char = char_list[idx]
            req[tag].append(char)
        pre_tag = tag
    req = dict(map(lambda t2: (
        t2[0],
        list(
        filter(lambda x: x.strip() ,"".join(t2[1]).split(sp_token))
        )
                         ), req.items()))
    return req

def ner_entity_type_predict(question, id_slice_num = 5):
    assert type(question) == type("")
    question = question.replace(" ", "")
    ner_df = from_text_to_final(
    " ".join(list(question)),
    tokenizer,
    zh_model,
    label_list
)
    assert ner_df.shape[0] == len(question) + 2
    ### [UNK] filling
    ner_df[0] = ["[CLS]"] + list(question) + ["[SEP]"]
    et_dict = for_loop_detect(ner_df.T.apply(lambda x: x.tolist(), axis = 1))

    et_id_dict = dict(
map(lambda t2: (
    t2[0], list(map(lambda x: np.asarray(x).reshape([-1]).tolist() ,zh_linker_entities(
        list(map(lambda x: [x], t2[1]))
    )[0]))
) ,et_dict.items())
)
    ori_entity_ids = et_id_dict.get("E-TAG", [])
    ori_type_ids = et_id_dict.get("T-TAG", [])

    return ori_entity_ids, ori_type_ids, et_dict

def keyword_rule_filter(question_rm_et ,query_prob_dict):
    assert type(question_rm_et) == type("")
    assert type(query_prob_dict) == type({})
    if not question_rm_et.strip():
        return query_prob_dict
    def how_many_edit_filter(query_prob_dict):
        if not query_prob_dict:
            return query_prob_dict

        if 'SELECT (COUNT(?obj) AS ?value ) { wd:E1 wdt:R1 ?obj }'\
         not in reduce(lambda a, b : a + b ,query_prob_dict.values()):
            return query_prob_dict

        #### contain rm
        rm_contain_list = ["多大"]
        if any(map(lambda x: x in question_rm_et, rm_contain_list)):
            return dict(filter(
                lambda t2: 'SELECT (COUNT(?obj) AS ?value ) { wd:E1 wdt:R1 ?obj }' not in t2[1],
                query_prob_dict.items()
            ))
        return query_prob_dict

    apply_func_list = [how_many_edit_filter]
    query_prob_dict = deepcopy(query_prob_dict)
    for f_func in apply_func_list:
        query_prob_dict = f_func(query_prob_dict)

    return query_prob_dict

def tmp_type_predict(question, question_rm_et, b_clf = b_tmp_clf, consider_tmp_prob = 0.2,
    show_query_prob_dict = False
):
    assert type(question) == type("")
    assert type(question_rm_et) == type("")
    prob_query_dict = tmp_from_text_to_final(question, cls_model = b_clf, sim_model = sim_model, return_query=True,
                      return_prob = True,
                      )
    assert type(prob_query_dict) == type({})
    query_prob_dict = dict(filter(lambda t2: t2[0] >= consider_tmp_prob, prob_query_dict.items()))

    if show_query_prob_dict:
        print("before :" ,query_prob_dict)

    query_prob_dict = keyword_rule_filter(question_rm_et ,query_prob_dict)

    if show_query_prob_dict:
        print("after :" ,query_prob_dict)

    query_list = list(map(lambda tt2: tt2[1] ,sorted(query_prob_dict.items(), key = lambda t2: t2[0], reverse = True)))
    query_list = reduce(lambda a, b: a + b, query_list) if query_list else []
    query_list = list(map(lambda x: x.strip(), query_list))

    tmp_func_7 = lambda el, pl: pd.Series(list(product(el, pl))).map(
            lambda ep: [(ep[0], ep[1], "")]
        ).explode().dropna().drop_duplicates().tolist()
    tmp_func_8 = lambda el, pl: pd.Series(list(product(el, pl))).map(
            lambda ep: [("", ep[1], ep[0])]
        ).explode().dropna().drop_duplicates().tolist()

    tmp_generate_t3_func_dict = {
        sparql_queries_df.iloc[0].to_dict()["query_template"]: tmp_func_7,
        sparql_queries_df.iloc[1].to_dict()["query_template"]: tmp_func_7,
        sparql_queries_df.iloc[2].to_dict()["query_template"]: tmp_func_7,
        sparql_queries_df.iloc[4].to_dict()["query_template"]: tmp_func_7,
        sparql_queries_df.iloc[7].to_dict()["query_template"]: tmp_func_7,
    }

    return dict(filter(lambda t2: t2[0] in query_list, tmp_generate_t3_func_dict.items()))

def property_df_rep_disambiguation(question ,property_df,
rep_col = "entity", property_col = "pid", order_col = "sum_score",
    use_emb = True
):
    print("do disamb in {}".format(question))
    property_df = drop_duplicates_of_every_df(property_df.sort_values(by = order_col, ascending = False))
    if not hasattr(property_df, "size") or property_df.size == 0:
        return property_df
    property_df_list = []
    for pid, df in property_df.groupby(property_col):
        s = df[rep_col][
            df[rep_col].map(bool)
        ]
        if not s.size:
            continue
        max_score = df[order_col].max()
        if use_emb:
            req_df = pd.DataFrame(s.map(
            lambda x: (x,
            ((max(
            perm_top_sort(question, x, sim_model, return_score = True)[1:]
            )) if len(x) >= 2 else synonyms.compare(x[0], question)) if type(x) == type([]) \
            else 1.0)
            ).values.tolist())
        else:
            req_df = pd.DataFrame(s.map(
            lambda x: (x,
            max(map(lambda y: synonyms.compare(y, question), x)) if type(x) == type([]) \
            else 1.0
            )
        ).values.tolist())
        req_df[2] = req_df[0].map(lambda x: -1 * len(x))
        rep_list = req_df.sort_values(by = [1, 2], ascending = False).iloc[:, 0].tolist()
        df = df[
            df[rep_col].map(
                lambda x: x in rep_list
            )
        ]
        df = df.iloc[:1, :]
        df[rep_col] = [rep_list] * len(df)
        df[order_col] = max_score
        property_df_list.append(df)
    req = pd.concat(property_df_list, axis = 0).sort_values(by = order_col, ascending = False)[
        ['pid',
 'entity',
 'multi_score',
 'score',
 'rank_score',
 'max_score',
 'sum_score']
    ]
    req["pid_in_cnt"] = req["pid"].map(
        lambda x: int(any(map(lambda y: y in question,
            list(filter(lambda xxx: xxx ,map(lambda xx: xx.strip() ,pid_zh_b_dict[x].split(" "))))\
             if x in pid_zh_b_dict else ["1-2-3-4-5-6-7-8-9-0"]
        )))
    )
    req = req.sort_values(by = ["pid_in_cnt", order_col], ascending = False)
    return req


def do_search(question, in_string_entity_overload_dict = {

}, do_property_disamb = True):
    ##### ner scope
    ori_entity_ids, ori_type_ids, et_dict = ner_entity_type_predict(question)

    question_rm_et = question
    assert type(et_dict) == type(dict())
    for k, v in et_dict.items():
        assert type(v) == type([])
        for vv in sorted(v, key = len, reverse = True):
            if vv:
                question_rm_et = question_rm_et.replace(vv, "")

    print("ori_entity_ids :")
    print(ori_entity_ids)

    overload_entityid_list = list(map(lambda tt2: tt2[1], filter(
    lambda t2: t2[0] in question
    , in_string_entity_overload_dict.items())))
    overload_entityid_list = list(set(overload_entityid_list).difference(set(
    reduce(lambda a, b: a + b, ori_entity_ids) if ori_entity_ids else ori_entity_ids
    ))) \
    if ori_entity_ids else overload_entityid_list

    if overload_entityid_list:
        assert overload_entityid_list[0].startswith("Q")

    print("overload_entityid_list :")
    print(overload_entityid_list)

    if overload_entityid_list and ori_entity_ids:
        need_extend_index = None
        zh_rep_list = list(map(
        lambda i:
        list(map(lambda x:
        search_entity_rep_by_lang_filter_in_db(x[1:] if x.startswith("XQ") else x, wiki_entity_db, "zh")
        , ori_entity_ids[i])),
        range(len(ori_entity_ids))))
        assert len(zh_rep_list) == len(ori_entity_ids)
        zh_rep_list = list(map(lambda x: list(set(reduce(lambda a, b: a + b, x))) if x else x,
        zh_rep_list))
        zh_rep_list = list(map(lambda x: x if x else [""], zh_rep_list))
        assert len(zh_rep_list) == len(ori_entity_ids)
        for ele in zh_rep_list:
            assert type(ele) == type([])
            assert ele

        in_string_entity_overload_dict_rev = dict(map(lambda t2: t2[::-1] ,
        in_string_entity_overload_dict.items()))
        for ele in overload_entityid_list:
            assert ele in in_string_entity_overload_dict_rev

        overload_entityid_index_dict = dict(map(
        lambda id:
        (id,

        sorted(map(
        lambda t2: (t2[0] ,

        max(map(lambda x:
        fuzz.ratio(x, in_string_entity_overload_dict_rev[id])
        ,t2[1]))

        )
        , enumerate(zh_rep_list)
        ),
        key = lambda tt2: tt2[1], reverse = True
        )[0][0] if sorted(map(
        lambda t2: (t2[0] ,

        max(map(lambda x:
        fuzz.ratio(x, in_string_entity_overload_dict_rev[id])
        ,t2[1]))

        )
        , enumerate(zh_rep_list)
        ),
        key = lambda tt2: tt2[1], reverse = True
        )[0][1] >= 50.0 else -1

        )
        , overload_entityid_list))

        print("overload_entityid_index_dict :")
        print(overload_entityid_index_dict)
        print("-" * 100)

        for k, v in overload_entityid_index_dict.items():
            assert k in overload_entityid_list
            assert v < len(ori_entity_ids) and v >= -1

        overload_entityid_index_dict_rev = defaultdict(list)
        for k, v in overload_entityid_index_dict.items():
            overload_entityid_index_dict_rev[v].append(k)

        ori_entity_ids_ = []
        for i, inner_list in enumerate(ori_entity_ids):
            ele = deepcopy(inner_list)
            if i in overload_entityid_index_dict_rev:
                assert type(overload_entityid_index_dict_rev[i]) == type([])
                ele = (overload_entityid_index_dict_rev[i] + ele)[:len(ele)]
            ori_entity_ids_.append(ele)

        if -1 in overload_entityid_index_dict_rev:
            ori_entity_ids_.append(overload_entityid_index_dict_rev[-1])

        assert len(ori_entity_ids_) >= len(ori_entity_ids) and len(ori_entity_ids_) in [0, 1, 2]
        ori_entity_ids = deepcopy(ori_entity_ids_)

    ori_entity_ids = list(filter(lambda x:
    list(filter(lambda y: y.startswith("Q") ,x))
     ,ori_entity_ids))

    if not ori_entity_ids:
        return None

    print("ori_entity_ids :")
    print(ori_entity_ids)

    ###### tmp classifier scope
    tmp_generate_t3_func_dict = tmp_type_predict(question, question_rm_et)

    tmp_generate_t3_func_dict = dict(filter(
        lambda t2: len(ori_entity_ids) != 1 if t2[0] in [
            'SELECT ?value WHERE { wd:E1 p:R1 ?s . ?s ps:R1 wd:E2 . ?s ?p ?value }'
        ] else True
    , tmp_generate_t3_func_dict.items()))

    if not tmp_generate_t3_func_dict:
        tmp_func_7 = lambda el, pl: pd.Series(list(product(el, pl))).map(
            lambda ep: [(ep[0], ep[1], "")]
        ).explode().dropna().drop_duplicates().tolist()

        tmp_generate_t3_func_dict = {
            sparql_queries_df.iloc[7].to_dict()["query_template"]: tmp_func_7,
        }

    query_list = sparql_queries_df["query_template"].tolist()
    assert len(set(tmp_generate_t3_func_dict.keys()).intersection(set(query_list))) == len(tmp_generate_t3_func_dict)

    #print("before ranking: ___")

    ###### ranking scope
    tmp_conclusion_dict = {}
    for query, func in tmp_generate_t3_func_dict.items():
        index = query_list.index(query)
        till = deepcopy(query_parser_bu(question,
        sparql_queries_df.iloc[index].to_dict(),
        sparql_queries_df.iloc[index]["entities_and_types_select"]
                                ,
                              ori_entity_ids, ori_type_ids))
        till = till_process_func(till)
        tmp_conclusion_dict[query] = till

    assert len(tmp_conclusion_dict) == len(tmp_generate_t3_func_dict)
    #return ori_entity_ids, tmp_conclusion_dict, tmp_generate_t3_func_dict

    #print("before choose tmp: ____")
    a, b, c = choose_tmp_by_ranking(
        question,
        list(map(
            lambda x: "http://www.wikidata.org/entity/{}".format(x)
            ,reduce(lambda a, b: a + b, ori_entity_ids)
            )),
        tmp_conclusion_dict,
        tmp_generate_t3_func_dict,
        aug_func = max,
    )
    if do_property_disamb:
        if not ori_type_ids:
            b = property_df_rep_disambiguation(question, b)
        else:
            type_id_zh_rep_list = list(map(lambda l:
            list(
            map(lambda id:
                search_entity_rep_by_lang_filter_in_db(id, wiki_entity_db, "zh") \
                if id.startswith("Q") else pid_zh_b_dict.get(id, "").split(" ")
                , l))
                                      , ori_type_ids)
                                     )

            print(type_id_zh_rep_list)
            type_id_zh_rep_list_ = []
            for inner_list in type_id_zh_rep_list:
                for ele in inner_list:
                    ele = list(filter(lambda x: x in question, ele))
                    for x in ele:
                        if x not in type_id_zh_rep_list_ and x in question:
                            type_id_zh_rep_list_.append(x)
            type_id_zh_rep_list = type_id_zh_rep_list_
            type_id_zh_rep_list = list(filter(lambda x: x.strip(), type_id_zh_rep_list))
            for ele in type_id_zh_rep_list:
                assert ele.strip()
            if not type_id_zh_rep_list:
                b = property_df_rep_disambiguation(question, b)
            else:
                b = property_df_rep_disambiguation(" ".join(type_id_zh_rep_list), b)

    return a, b, c

br_cls = b_rel_ranker.rfr_cls

def ner_entity_type_predict_only(question):
    assert type(question) == type("")
    question = question.replace(" ", "")
    ner_df = from_text_to_final(
    " ".join(list(question)),
    tokenizer,
    zh_model,
    label_list
)
    assert ner_df.shape[0] == len(question) + 2
    ### [UNK] filling
    ner_df[0] = ["[CLS]"] + list(question) + ["[SEP]"]
    et_dict = for_loop_detect(ner_df.T.apply(lambda x: x.tolist(), axis = 1))
    return et_dict

'''
rep = requests.post(
    url = "http://localhost:8855/extract_et",
    data = {
        "question": "哈利波特的作者是谁？"
    }
)
json.loads(rep.content.decode())
'''
@csrf_exempt
def extract_et(request):
    assert request.method == "POST"
    post_data = request.POST
    question = post_data["question"]
    assert type(question) == type("")
    #question = "宁波在哪个省？"
    #abc = do_search(question)
    et_dict = ner_entity_type_predict_only(question)
    assert type(et_dict) == type({})
    return HttpResponse(json.dumps(et_dict))

'''
rep = requests.post(
    url = "http://localhost:8855/property_score",
    data = {
        "question": "哈利波特的作者是谁？"
    }
)
json.loads(rep.content.decode())
'''
@csrf_exempt
def property_score(request):
    assert request.method == "POST"
    post_data = request.POST
    question = post_data["question"]
    assert type(question) == type("")
    ranking_df = br_cls.produce_rank_df(question,  br_cls.all_cate_list)
    assert hasattr(ranking_df, "shape")
    assert ranking_df.shape[1] == 2
    req = ranking_df[["cate", "prob"]].values.tolist()
    req = json.loads(json.dumps(req))
    return HttpResponse(json.dumps(req))

'''
rep = requests.post(
    url = "http://localhost:8855/zh_entity_link",
    data = {
        "entity_str": "李芳果"
    }
)
json.loads(rep.content.decode())
'''
@csrf_exempt
def zh_entity_link(request):
    assert request.method == "POST"
    post_data = request.POST
    entity_str = post_data["entity_str"]
    assert type(entity_str) == type("")
    output = zh_linker_entities(
        [[entity_str]]
    )
    assert type(output) == type((1,))
    pid_list = np.asarray(output[0]).reshape([-1]).tolist()
    pid_list = list(filter(lambda x: type(x) == type("") and x.startswith("Q"), pid_list))
    if not pid_list:
        return HttpResponse(json.dumps(pid_list))
    pid_list = list(map(
    lambda qid: (qid, search_entity_rep_by_lang_filter_in_db(qid, wiki_entity_db, lang = "zh"))
    , pid_list))
    return HttpResponse(json.dumps(pid_list))

'''
rep = requests.post(
    url = "http://localhost:8855/kbqa",
    data = {
        "question": "指环王的作者是谁？"
    }
)
json.loads(rep.content.decode())
'''
@csrf_exempt
def kbqa(request):
    assert request.method == "POST"
    post_data = request.POST
    question = post_data["question"]
    assert type(question) == type("")
    #question = "宁波在哪个省？"
    #abc = do_search(question)

    try:
        abc = do_search(question)
    except:
        abc = None
    '''
    print("abc :")
    print(abc)
    print("type of abc :", type(abc))
    '''
    assert abc is None or type(abc) == type((1,))
    if abc is not None:
        assert len(abc) == 3
    else:
        return HttpResponse(json.dumps(
            {"output": "No Answer"}
        ))
    a, b, c = abc

    assert type(b) == type(pd.DataFrame())
    if b.size == 0:
        return HttpResponse(json.dumps(
            {"output": "No Answer"}
        ))
    assert "entity" in b.columns.tolist()
    e_l = b["entity"].map(str).values.tolist()
    assert type(e_l) == type([])
    output = {
        "output": e_l
    }
    return HttpResponse(json.dumps(output))
