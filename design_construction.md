### A Review on knowledge base question answer systems

There is a point of view that distinguish different nlp question answer systems, that may be the
organization of answers. The question answer system is a system that transform (or map) natural language
to the different hierarchy of answer space. The answer space can be both probability space 、
distance space or discrete space. The former two space can be arrived by nlu classifier or distance
statistics. The last one is this project focus on and related topic as TableQA(natural language to SQL)
is a similar topic where this project focus on "translate" natural language to SPARQL query.


### A Summary on natural language to SPARQL query

Mainly two kinds of natural language to SPARQL query methods. One is a reconstruct method that try to
decide a suitable base form of SPARQL query and fill it by correspond semantic components extract from
natural language, The other is a transform method that make this process like a translation between
natural language and SPARQL query. The above discuss also compliant with TableQA and other natural language
to query topics.


### Open Source Projects

Nowadays(2022 and before) ,In open source community, [DeepPavlov](https://github.com/deeppavlov/DeepPavlov)
and [Haystack](https://github.com/deepset-ai/haystack/) are two projects that implement KBQA use the above two
different methods. And this project focus on adjust the former into Chinese language domain.


### Construction of DeepPavlov KBQA and some tips to overload

KBQA in DeepPavlov have many kind of implementations. Such as decompose natural language by
Universal Dependencies parser (that use tree decomposition to extract semantic sub construction)
or Use NER to extract entities and properties from natural language directly. Than one can use
them to full fill a SPARQL query template (choose by a template classifier), use a batch of query
perform search on Knowledge Base that will retrieve many candidates. Then perform a ranking on them.
The language components that can be replaced to make it satisfy Chinese domain are include entity extraction、 entity linking 、query template classifier、candidates ranking components. Entity linking reconstruction can be done by perform Chinese word segmentation on Chinese entity representations which associate the English one by wikidataId. Other three components are mainly about translation (and the corpus translation about entity extraction may be skillful)

### Project Construction
Following is the function of scripts:<br>

ner_model.py : entity extraction component<br>
tmp_classifier.py: query template classifier<br>
ranker.py: candidates ranking component<br>
kbqa_step.py: main script that perform the KBQA task

The main closure of above scripts have some examples that you can try yourself.

In the mysite folder, is a server build by Django, you can use run.sh to start it
and use request to use it.
