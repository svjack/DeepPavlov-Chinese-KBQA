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

abcde_dict = {
    "a":  [
            'SELECT ?ent WHERE { wd:E1 wdt:R1 ?ent } ',
        ],
    "b": [
        'SELECT ?value WHERE { wd:E1 p:R1 ?s . ?s ps:R1 wd:E2 . ?s ?p ?value }',
        ],
    "c": [
        'SELECT ?obj WHERE { wd:E1 p:R1 ?s . ?s ps:R1 ?obj . ?s ?p ?x filter(contains(?x, N)) }'
    ,
],
    "d": [
            'SELECT ?value WHERE { wd:E1 p:R1 ?s . ?s ps:R1 ?x filter(contains(?x, N)) . ?s ?p ?value }',
    ],
    "e": [
       'SELECT (COUNT(?obj) AS ?value ) { wd:E1 wdt:R1 ?obj }',
    ]
}


def load_pickle(path):
    assert os.path.exists(path)
    with open(path, "rb") as f:
        return pkl.load(f)

def tmp_from_text_to_final(str_, cls_model, sim_model,
    return_query = False, return_prob = False
):
    assert type(str_) == type("")
    #assert type(class_to_query_mapping) == type({})
    assert hasattr(cls_model, "predict")
    result = sim_model.encode([str_])
    X = np.asarray(result).reshape([1, 768])
    if return_prob:
        assert hasattr(cls_model, "predict_proba")
        y = cls_model.predict_proba(X)
        y = y.reshape([-1])
    else:
        y = cls_model.predict(X)
    if return_query:
        #return class_to_query_mapping[y[0]]
        #print(y[0])
        #label = dict(map(lambda x: x[::-1], label_Class_dict.items()))[y[0]]
        #print(y)
        if not return_prob:
            label = y[0]
            return abcde_dict[label]
        else:
            return dict(map(lambda t2:
                       (y[cls_model.classes_.tolist().index(t2[0])].reshape([-1])[0], t2[1])
                       , abcde_dict.items()))
    return y[0]

#b_tmp_clf = load_pickle("tmp_cls/tmp_bag_mlp.pkl")
b_tmp_clf = load_pickle(os.path.join(main_path ,"tmp_cls/tmp_bag_mlp.pkl"))

if __name__ == "__main__":
    sim_model = SentenceTransformer('LaBSE')
    sim_model.pool = None

    tmp_from_text_to_final("宁波在哪个省？", cls_model = b_tmp_clf, sim_model = sim_model,
    return_query=True,
                  return_prob=True
                  )

    tmp_from_text_to_final("美国的通货是什么？", cls_model = b_tmp_clf, sim_model = sim_model,
    return_query=True,
                  return_prob=True
                  )

    tmp_from_text_to_final("埃尔达尔·梁赞诺夫出生在薩馬拉的时候他出生在哪个国家？", cls_model = b_tmp_clf, sim_model = sim_model,
    return_query=True,
                  return_prob=True
                  )
