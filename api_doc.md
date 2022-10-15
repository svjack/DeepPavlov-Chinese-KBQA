### Function Documentation
<b>ner_model.py</b>:<br> A self-trained NER model that extract entities and properties from Chinese questions.
It use adapter-hub's [adapter-transformers](https://github.com/adapter-hub/adapter-transformers)
on NER downstream task.
The E-TAG indicate entity type and T-TAG as properties.

<b>tmp_classifier.py</b>:<br> A self-trained BaggingClassifier that use MLP as BaseModel to classify Chinese
question into 5 classes (defined in abcde_dict), with a multilanguage encoder (named LaBSE) to
encode the text into dense space.
Use Bagging because the 5 classes is unbalanced (also with some sampling)

<b>ranker.py</b>:<br> A self-trained BaggingClassifier that use MLP as BaseModel to classify Chinese
question into 2 classes. This task similar with CrossEncoder in [sentence-transformers](https://github.com/UKPLab/sentence-transformers), make pair input as (chinese_question, property_representation), train a 0-1 classifier to find the highest score pair, that the property_representation represent the question reasonable. This may indicate the evidence about
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

<b>syn_sim_on_list</b>:<br> find the similar text from a collection of list compared with another text by distance defined by [synonyms](https://github.com/chatopera/Synonyms) (text only maintain
Chinese parts)

<b>t3_statement_ranking, choose_tmp_by_ranking</b>:<br> use ranker to find a reasonable s p o ranking conclusion between Chinese question and the many s p o collections.

<b>till_process_func</b>:<br> some SPARQL part have some decorate such as 'FILTER (?x = a ).' so the s p o will be expand to s p o f. This function filter out the part we only careful.

<b>fill_str, for_loop_detect</b>:<br> decode BIO style conclusion from NER model to a dictionary with [E-TAG T-TAG O-TAG] as keys and list of elements as values.

<b>ner_entity_type_predict</b>:<br> use adapter-transfomers to extract entities and properties of a Chinese question.

<b>keyword_rule_filter</b>:<br> a rule based fix on the output of tmp_classifier. In the definition, every question with "多大" as its sub-span will drop the "COUNT" style SPARQL template.

<b>tmp_type_predict</b>:<br> use tmp_classifier to classify Chinese question into 5 templates defined in abcde_dict.

<b>property_df_rep_disambiguation</b>:<br> disambiguate different properties on question.

<b>do_search</b>:<br> The main function that input the Chinese question and output the query conclusion from wikidata hdt Knowledge Base.
