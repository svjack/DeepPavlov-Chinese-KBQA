#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ranker import *


# In[2]:


sim_model = SentenceTransformer('LaBSE')
sim_model.pool = None

br_cls = RFR(b_clf,
        all_cate_list=list(pid_zh_b_dict.values()),
       sim_model=sim_model
       )


# In[134]:


property_info_df = load_pickle("property_info_df.pkl")


# In[135]:


property_info_df = property_info_df[
    property_info_df["pid"].isin(list(pid_zh_b_dict.keys()))
]


# In[136]:


property_info_df["zh_info_str"] = property_info_df["pid"].map(
    lambda x: pid_zh_b_dict[x]
)


# In[137]:


property_info_df["en_info"] = property_info_df["en_info"].map(
    lambda x: [sorted(x, key = len)[0]] if x else []
)


# In[138]:


property_info_df["zh_info"] = property_info_df["zh_info_str"].map(
    lambda x: list(map(lambda y: y.strip(), x.split(" ")))
)


# In[139]:


property_info_df = property_info_df[
    property_info_df.apply(lambda x: x["en_info"] and x["zh_info"], axis = 1).map(bool)
]


# In[142]:


####property_info_df


# In[ ]:





# In[95]:


br_cls.produce_rank_df("美国的货币是什么？",  br_cls.all_cate_list).head(2)


# In[100]:


pd.merge(
    br_cls.produce_rank_df("美国的货币是什么？",  br_cls.all_cate_list).head(2), 
    property_info_df,
    left_on = "cate", right_on = "zh_info"
)


# In[ ]:


property_info_df_str = property_info_df.applymap(
    lambda x: if type(x) == type()
)


# In[7]:


from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type="mt5", model_name="google/mt5-base")

model.load_model(
    "mt5",
    "/Users/svjack/temp/kb_aug/model/t5_gen_outputs/nq-simplet5-epoch-2-train-loss-0.0072-val-loss-0.0045",
    use_gpu = False
)


# In[9]:


import sqlite_utils


# In[10]:


assert os.path.exists("kbqa-explore/multi_lang_kb_dict.db")
wiki_entity_db = sqlite_utils.Database("kbqa-explore/multi_lang_kb_dict.db")
assert "en_zh_so_search" in wiki_entity_db.table_names()


# In[11]:


sent = "碳酸钙的副产品是什么? 在温度为1000的时候?"


# In[101]:


from fuzzywuzzy import fuzz


# In[35]:


import jieba.posseg as posseg
def unzip_string(x, size = 2):
    if len(x) <= size:
        return [x]
    req = []
    for i in range(len(x) - size + 1):
        req.append(x[i: i + size])
    return req


# In[86]:


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


# In[147]:


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


# In[148]:


a, b = entity_property_simplify_extraction("宁波在哪个省？", br_cls)


# In[149]:


a


# In[150]:


b


# In[151]:


"宁波在哪个省" + " * " + "{'Ningbo': '宁波'}" + "|" + "{'in': '所在省'}"


# In[162]:


model.predict(
    "宁波在哪个省" + " * " + "{'Ningbo': '宁波'}" + "|" + "{'in': '所在省'}"
)


# In[163]:


model.predict(
    "宁波在哪个省" + " * " + "{'宁波': '宁波'}" + "|" + "{'所在省': '所在省'}"
)


# In[164]:


model.predict(
    "宁波在哪个省" + " * " + "{'所在省': '所在省'}" + "|" + "{'宁波': '宁波'}"
)


# In[ ]:





# In[165]:


model.predict(
    "碳酸钙的副产品是什么? 在温度为1000的时候? * {'by-product': '副产物'}|{'temperature': '室温'}|{'calcium carbonate': '碳酸钙'}"
)


# In[166]:


model.predict(
    "碳酸钙的副产品是什么? 在温度为1000的时候? * {'副产物': '副产物'}|{'室温': '室温'}|{'碳酸钙': '碳酸钙'}"
)


# In[169]:


model.predict(
    "哪个省管辖丰县？"
)


# In[ ]:





# In[168]:


model.predict(
    "碳酸钙的副产品在温度为1000的时候是什么? * {'副产物': '副产物'}|{'室温': '室温'}|{'碳酸钙': '碳酸钙'}"
)


# In[ ]:





# In[167]:


model.predict(
    "碳酸钙的副产品是什么? 在温度为1000的时候? * {'碳酸钙': '碳酸钙'}|{'副产物': '副产物'}|{'室温': '室温'}"
)


# In[ ]:





# In[154]:


model.predict(
    "宁波在哪个省" + " * " + "{'Ningbo': '宁波'}" + "|" + "{'in': '所在省'}" + "|" + "{'département': '省'}"
)


# In[155]:


model.predict(
    "宁波在哪个省"
)


# In[156]:


model.predict(
    "碳酸钙的副产品是什么? 在温度为1000的时候?"
)


# In[157]:


model.predict(
    "哪个职业是乔治布莱尔的人生巅峰?"
)


# In[158]:


model.predict(
    "碳酸钙的副产品是什么?"
)


# In[ ]:





# In[ ]:





# In[ ]:


'select distinct?answer where {wd:Ningbo wdt:in ?answer}'


# In[ ]:


model.predict(
    "宁波在哪个省" + " * " + "{'Ningbo': '宁波'}" + "|" + "{'in': '所在省'}" + "|" + 
)


# In[ ]:


model.predict(
    "宁波在哪个省" + " * " + "{'Ningbo': '宁波'}" + "|" + "{'in': '所在省'}"
)


# In[ ]:





# In[115]:


property_info_df[
    property_info_df["pid"] == "P131"
]


# In[ ]:





# In[88]:


iter_sent_on_db(sent)


# In[89]:


iter_sent_on_db("宁波在哪个省？")


# In[ ]:





# In[ ]:





# In[90]:


iter_sent_on_db("美国的货币是什么？")


# In[91]:


iter_sent_on_db("川普什么时候出生？")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


####


# In[ ]:





# In[65]:


zh_str = '碳酸'
g = wiki_entity_db.query("select s from en_zh_so_search where o like '%{}%'".format(zh_str))
l = list(g)


# In[66]:


len(l)


# In[67]:


pd.Series(l)


# In[18]:


zh_str = '碳酸'
g = wiki_entity_db.query("select * from en_zh_so_search where o == '{}'".format(zh_str))
l = list(g)


# In[19]:


len(l)


# In[159]:


model.predict(
    "哪个职业是乔治布莱尔的人生巅峰? * {'occupation': '职业'}|{'highlight moment': '高光时刻'}｜{'George Black': '乔治布莱克'}"
)


# In[160]:


model.predict(
    "哪个职业是乔治布莱尔的人生巅峰? * {'status': '身份'}|{'moment': '时刻'}｜{'Black': '布莱克'}"
)


# In[161]:


model.predict(
    "哪个职业是乔治布莱尔的人生巅峰? * {'身份': '身份'}|{'时刻': '时刻'}｜{'布莱克': '布莱克'}"
)


# In[ ]:


model.predict(
    "哪个职业是乔治布莱尔的人生巅峰? * {'身份': '身份'}|{'时刻': '时刻'}｜{'布莱克': '布莱克'}"
)

