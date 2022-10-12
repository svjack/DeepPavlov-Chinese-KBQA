/Users/svjack/opt/anaconda3/envs/py38/bin/python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -language zh \
 -threads 10 -input collection_json \
 -index lucene-index-clean -storePositions -storeDocvectors -storeContents
