#!/usr/bin/env python
# coding: utf-8

# In[1]:
from ranker import *


sim_model = SentenceTransformer('LaBSE')
sim_model.pool = None

br_cls = RFR(b_clf,
        all_cate_list=list(pid_zh_b_dict.values()),
       sim_model=sim_model
       )


property_info_df = load_pickle("property_info_df.pkl")
property_info_df = property_info_df[
    property_info_df["pid"].isin(list(pid_zh_b_dict.keys()))
]
property_info_df["zh_info_str"] = property_info_df["pid"].map(
    lambda x: pid_zh_b_dict[x]
)
property_info_df["en_info"] = property_info_df["en_info"].map(
    lambda x: [sorted(x, key = len)[0]] if x else []
)
property_info_df["zh_info"] = property_info_df["zh_info_str"].map(
    lambda x: list(map(lambda y: y.strip(), x.split(" ")))
)
property_info_df = property_info_df[
    property_info_df.apply(lambda x: x["en_info"] and x["zh_info"], axis = 1).map(bool)
]


###br_cls.produce_rank_df("美国的货币是什么？",  br_cls.all_cate_list).head(2)

from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type="mt5", model_name="google/mt5-base")

model.load_model(
    "mt5",
    "/Users/svjack/temp/kb_aug/model/t5_gen_outputs/nq-simplet5-epoch-2-train-loss-0.0072-val-loss-0.0045",
    use_gpu = False
)


import sqlite_utils

assert os.path.exists("kbqa-explore/multi_lang_kb_dict.db")
wiki_entity_db = sqlite_utils.Database("kbqa-explore/multi_lang_kb_dict.db")
assert "en_zh_so_search" in wiki_entity_db.table_names()

sent = "碳酸钙的副产品是什么? 在温度为1000的时候?"

from fuzzywuzzy import fuzz
import jieba.posseg as posseg
def unzip_string(x, size = 2):
    if len(x) <= size:
        return [x]
    req = []
    for i in range(len(x) - size + 1):
        req.append(x[i: i + size])
    return req

def iter_sent_on_db(sent, db = wiki_entity_db, maintain_flag = ["n", "v"], refine_token_size = 2,
                   rm_keys = ["是"], take_one = True, rp_eng = True
                   ):
    sent_c = set(map(lambda tt2: tt2[0] ,filter(lambda t2:
                    any(map(lambda f: t2[1].startswith(f), maintain_flag))
                    ,map(lambda x: (x.word, x.flag) ,posseg.lcut(sent)))))

    sent_c = pd.Series(list(sent_c)).map(
        lambda x: unzip_string(x, refine_token_size)
    ).explode().dropna().drop_duplicates().tolist()

    if not sent_c:
        return []
    l = list(map(
        lambda x: (x,
        list(map(lambda y: y["o"] ,
        wiki_entity_db.query("select * from en_zh_so_search where o like '%{}%'".format(x)))))
        , sent_c))
    #return l
    l = list(map(lambda t2: (
        t2[0],
        sorted(set(filter(lambda x: x in sent, t2[1])), key = len, reverse = True)
    ), l))
    ll = []
    for k, v in l:
        if v and k not in rm_keys:
            vvv = []
            #print(v)
            for vv in v:
                assert type(vv) == type("")
                vl = list(wiki_entity_db.query("select s from en_zh_so_search where o = '{}'".format(vv)))
                sl = list(map(lambda x: x["s"], vl))
                if take_one:
                    sl = [sorted(sl, key = lambda x:int(x[1:]))[0]]
                    if rp_eng:
                        sl = list(wiki_entity_db.query("select o from en_zh_so_search where s = '{}' and lang = 'en'".format(sl[0])))
                        sl = list(map(lambda x: x["o"], sl))
                vvv.extend(sl)
                #print(vvv)
            ll.append((k, vvv))
    return dict(ll)

def entity_property_simplify_extraction(
    sent, br_cls, property_topk = 3
):
    print("fit entity")
    entity_part = iter_sent_on_db(sent)
    entity_part = dict(
        map(lambda t2: (t2[0], [sorted(t2[1], key = len)[0]] if t2[1] else []) ,entity_part.items())
    )
    print("pred prop")

    property_part = br_cls.produce_rank_df(sent,  br_cls.all_cate_list).head(property_topk)
    #print("0 :")
    #print(property_part)
    property_part = pd.merge(
    property_part,
    property_info_df,
    left_on = "cate", right_on = "zh_info_str"
)[["pid", "zh_info", "en_info", "prob"]]
    #print("1 :")
    #print(property_part)
    property_part["en_info"] = property_part["en_info"].map(
        lambda x: [sorted(x, key = len)[0]]
    )
    property_part["zh_info"] = property_part["zh_info"].map(
        lambda x: [sorted(x, key = lambda y: fuzz.ratio(y, sent), reverse = True)[0]]
    )
    return entity_part, property_part

a, b = entity_property_simplify_extraction("宁波在哪个省？", br_cls)
