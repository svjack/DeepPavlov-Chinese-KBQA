from conf import *
#main_path = "/Volumes/TOSHIBA EXT/temp/kbqa_portable_prj"

import pandas as pd
import pickle as pkl
import numpy as np
import os

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier

from sentence_transformers import SentenceTransformer
import json

def load_pickle(path):
    assert os.path.exists(path)
    with open(path, "rb") as f:
        return pkl.load(f)

class RFR(object):
    def __init__(self, clf, all_cate_list, sim_model):
        assert hasattr(clf, "fit")
        assert type(all_cate_list) == type([])
        self.clf = clf
        self.all_cate_list = all_cate_list
        self.sim_model = sim_model
        self.all_cate_emb_dict = {}

        self.produce_all_cate_emb()

        self.all_cate_emb_df = pd.DataFrame(self.all_cate_emb_dict.items(), columns = ["rank_key", "rank_key_emb"])

    def produce_all_cate_emb(self):
        arr = self.sim_model.encode(self.all_cate_list,
            show_progress_bar = True
        )
        assert len(arr) == len(self.all_cate_list)
        ele = self.all_cate_list
        d = {}
        for i in range(len(ele)):
            k = ele[i]
            v = arr[i]
            d[k] = v
        self.all_cate_emb_dict = d

    def emb_one_sent(self, sent):
        assert type(sent) == type("")
        req = self.sim_model.encode([sent])
        if hasattr(req, "numpy"):
            req = req.numpy()
        return req

    def produce_rank_df(self, sent, filter_list = []):
        sent_emb = self.emb_one_sent(sent)
        sent_emb_l = np.asarray(sent_emb).reshape([-1]).tolist()
        assert len(sent_emb_l) == 768
        req = self.all_cate_emb_df.copy()
        req = req[
            req["rank_key"].isin(filter_list)
        ]

        req["x"] = req.apply(
            lambda s:
            sent_emb_l + s["rank_key_emb"].reshape([-1]).tolist()
            , axis = 1
        )
        x = np.asarray(req["x"].values.tolist()).reshape([-1, 768 * 2])
        assert len(x.shape) == 2
        assert x.shape[0] == len(req)

        pred = self.clf.predict_proba(x)
        assert len(pred.shape) == 2
        pred = pred[:, 1]
        assert len(pred) == len(req)

        req = pd.concat([
            pd.Series(pred), pd.Series(req["rank_key"].values.tolist())
        ], axis = 1)

        req.columns = ["prob", "cate"]
        return req.sort_values(by = "prob", ascending = False)

with open(os.path.join(main_path ,"ranker_cls/pid_zh_b_dict.json"), "r") as f:
    pid_zh_b_dict = json.load(f)

b_clf = load_pickle(os.path.join(main_path ,"ranker_cls/ranking_bag_mlp.pkl"))
if __name__ == "__main__":
    sim_model = SentenceTransformer('LaBSE')
    sim_model.pool = None

    br_cls = RFR(b_clf,
        all_cate_list=list(pid_zh_b_dict.values()),
       sim_model=sim_model
       )

    br_cls.produce_rank_df("宁波在哪个省？",  br_cls.all_cate_list)
    br_cls.produce_rank_df("美国的货币是什么？",  br_cls.all_cate_list)
    br_cls.produce_rank_df("埃尔达尔·梁赞诺夫出生在薩馬拉的时候他出生在哪个国家？",  br_cls.all_cate_list)
