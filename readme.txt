
#### sqlite table to django model
g = wiki_entity_db.query("select * from en_zh_so_search where s = '{}' and lang = '{}'".format(id, lang))

'''
from django.db import models
from django.contrib.postgres.fields import ArrayField

# Create your models here.

class Embedding(models.Model):
    sent = models.CharField(max_length=200)
    emb = ArrayField(models.FloatField())

class Translation(models.Model):
    sent = models.CharField(max_length=2000)
    trans = models.CharField(max_length=2000)

class Valid(models.Model):
    valid_string = models.CharField(max_length=2000)

class en_zh_so(models.Model):
    '''
    {'s': 'Q1', 'o': '"universe"', 'lang': 'en'}
    dump to local:
    nohup sqlite-utils multi_lang_kb_dict.db "select row_number() OVER(ORDER BY s) ,* from en_zh_so" --csv --no-headers > db_dump.csv &

    import pandas as pd
    from tqdm import tqdm
    from polls.models import en_zh_so

    db_dump_df = pd.read_csv("mysite_wiki/db_dump.csv", header = None)
    dmodel = en_zh_so
    for i, r in tqdm(db_dump_df.iterrows()):
        d = {
        "s": r.iloc[1],
        "o": r.iloc[2],
        "lang": r.iloc[3]
        }
        dm = dmodel(**d)
        dm.save()
    '''
    s = models.CharField(max_length=2000)
    o = models.CharField(max_length=2000)
    lang = models.CharField(max_length=2000)

'''

nohup sqlite-utils multi_lang_kb_dict.db "select row_number() OVER(ORDER BY s) ,* from en_zh_so" --csv --no-headers > db_dump.csv &
nohup sqlite-utils multi_lang_kb_dict.db "select row_number() OVER(ORDER BY s) ,* from en_zh_so_search" --csv --no-headers > db_dump.csv &
