<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">DeepPavlov-Chinese-KBQA</h3>

  <p align="center">
   		Chinese Knowledge Question Answer System based on DeepPavlov
    <br />
  </p>
</p>

### Brief introduction

DeepPavlov is an open-source conversational AI library built on TensorFlow, Keras and PyTorch, which
has a Knowledge Base Question Answering model that support perform knowledge base qa on both English and
Russian.
This project focus on reconstruct a KBQA system on Chinese by replace corresponding semantic parsing
components into Chinese version and finetuned on Chinese domain. Below are some details.


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
Following is the function of scripts:

ner_model.py : entity extraction component
tmp_classifier.py: query template classifier
ranker.py: candidates ranking component
kbqa_step.py: main script that perform the KBQA task

The main closure of above scripts have some examples that you can try yourself.

In the mysite folder, is a server build by Django, you can use run.sh to start it
and use request to use it.

### Function Documentation
<b>ner_model.py</b>:<br> A self-trained NER model that extract entities and properties from Chinese questions.
It use adapter-hub's [adapter-transformers](https://github.com/adapter-hub/adapter-transformers)
on NER downstream task.
The E-TAG indicate entity type and T-TAG as properties.

<b>tmp_classifier.py</b>:<br> A self-trained BaggingClassifier that use MLP as BaseModel to classify Chinese
question into 5 classes (defined in abcde_dict), with a multilanguage encoder (named LaBSE) to
encoding the text into dense space.
Use Bagging because the 5 classes is unbalanced (also with some sampling)

<b>ranker.py</b>:<br> A self-trained BaggingClassifier that use MLP as BaseModel to classify Chinese
question into 2 classes. This task similar with CrossEncoder in [sentence-transformers](https://github.com/UKPLab/sentence-transformers), make may pair input as (chinese_question, property_representation), train a 0-1 classifier to find the highest score pair, that the property_representation represent the question reasonable. This many indicate the evidence about
the answer that satisfy the ask intent.

<b>kbqa_step.py</b>:<br> Main script that perform the KBQA task.

function definitions:

<b>search_entity_rep_by_lang_filter_in_db</b>:<br> find language representations of a wikidataId by setting the language flag (support en and zh) in a pre-build sqlite database, this DB can be analogy to the translate dictionary of entities cross English and Chinese.

<b>Zh_Rel_Ranker</b>:<br> definition of above ranker object

<b>query_parser_bu, find_top_rels_bu</b>:<br> main part of query process in DeepPavlov

<b>t3_statement_df</b>:<br> perform a SPARQL query inquiry on the wikidata hdt file and represent the
  conclusion as a [n, 3] shaped pandas dataframe (with columns named with s p o, where s p o
    is the basic Triad collection in Knowledge Base)

<b>fix_o</b>:<br> a toolkit that fix some problem when transform the stream made by hdt query iterator's o field when collect this stream to a local Ntriple file.

<b>py_dumpNtriple</b>:<br> transform on row of s p o made by hdt query iterator to Ntriple file format.

<b>one_part_g_producer</b>:<br> init a knowledge graph object with the help of rdflib

<b>drop_duplicates_by_col</b>:<br> a toolkit that drop the duplicates of a pandas dataframe by unify the value of one column

<b>drop_duplicates_of_every_df</b>:<br> a toolkit that drop the duplicates of a pandas dataframe of any dtypes (this function is useful when some cells in dataframe not have hashcode : e.x. List)

<b>search_triples_with_parse</b>:<br> perform a SPARQL query inquiry on the wikidata hdt file

<b>perm_top_sort</b>:<br> find the similar text from a collection of list compared with another text by cos distance between SentenceTransformer text encodings.

<b>syn_sim_on_list</b>:<br> find the similar text from a collection of list compared with another text by distance defined by (synonyms)[https://github.com/chatopera/Synonyms] (text only maintain
Chinese parts)

<b>t3_statement_ranking, choose_tmp_by_ranking</b>:<br> use ranker to find a reasonable s p o ranking conclusion between Chinese question and the many s p o collections.

<b>till_process_func</b>:<br> some SPARQL part have some decorate such as 'FILTER (?x = a ).' so the s p o will be expand to s p o f. This function filter out the part we only careful.

<b>fill_str, for_loop_detect</b>:<br> decode BIO style conclusion from NER model to a dictionary with [E-TAG T-TAG O-TAG] as keys and list of elements as values.

<b>ner_entity_type_predict</b>:<br> use adapter-transfomers to extract entities and properties of a Chinese question.

<b>keyword_rule_filter</b>:<br> a rule based fix on the output of tmp_classifier. In the definition, every question with "多大" as its sub-span will drop the "COUNT" style SPARQL template.

<b>tmp_type_predict</b>:<br> use tmp_classifier to classify Chinese question into 5 templates defined in abcde_dict.

<b>property_df_rep_disambiguation</b>:<br> disambiguate different properties on question.

<b>do_search</b>:<br> The main function that input the Chinese question and output the query conclusion from wikidata hdt Knowledge Base.
