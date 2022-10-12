conda create -n kbqa_env python==3.7

conda activate kbqa_env

pip install pybind11

pip install -r requirements.txt

pip install deeppavlov

pip install editdistance

pip install tensorflow==1.15.2

pip install -r bert_dp.txt

pip install spacy==2.3.3

python -m spacy download en_core_web_sm

pip install whapi

pip install faiss

pip install adapter_transformers==2.1.2

pip install datasets

pip install -r sortedcontainers.txt
