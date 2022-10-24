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

[DeepPavlov](https://github.com/deeppavlov/DeepPavlov) is an open-source conversational AI library built on TensorFlow, Keras and PyTorch, which
has a Knowledge Base Question Answering model that support perform knowledge base qa on both English and
Russian.
This project focus on reconstruct a KBQA system on Chinese by replace corresponding semantic parsing
components into Chinese version and finetuned on Chinese domain.


### Installation
Refer to INSTALL.sh to install the environment, make sure that you can run the KBQA of the original DeepPavlov project.
And because the installation may be difficult i have build a docker image. with the pretrained models located in model, sel_ner, tmp_cls, ranker_cls .If someone need, you can send a mail to ehangzhou@outlook.com to get them.
The wikidata Knowledge Base hdt file can get from me or the [rdfhdt](https://www.rdfhdt.org/datasets/) and make sure the version you download contain Chinese part.(some only have English part).

Pretrained models below should be placed in the correspond location. 
files below kbqa-explore path can be download following [installtion of LC-QuAD-augmentation-toolkit](https://github.com/svjack/LC-QuAD-augmentation-toolkit) others can email me to get.
```yml
model: #### LaBSE embedding
- LaBSE_local  

sel_ner: #### Entity Extraction
- adapter_config.json
- ner_data_args.pkl
- pytorch_model_head.bin
- head_config.json
- pytorch_adapter.bin

tmp_cls: #### Template classifier
- tmp_bag_mlp.pkl

ranker_cls: #### Ranking model
- pid_zh_b_dict.json
- ranking_bag_mlp.pkl

kbqa-explore: #### Knowledge Base and translation dictionary
- multi_lang_kb_dict.db
- wikidata.hdt
- linker_entities.pkl
```

### KBQA Usage
You can run kbqa_step.py directly (call do_search in main closure) or cd into mysite and activate the environment you install KBQA (in INSTALL.sh)
```bash
conda activate kbqa_env
bash run.sh
```
this will run a server in 8855 ,then you can use it  to retrieve conclusion like:<br/>
<br/>
<b>Example 1:</b>
```python
import requests
import json
rep = requests.post(
    url = "http://localhost:8855/kbqa",
    data = {
        "question": "指环王的作者是谁？"
    }
)
json.loads(rep.content.decode())["output"][:3]
```
this will output
```json
["[['J·R·R·托尔金', 'J·R·R·托爾金', 'J·R·R·託爾金', '托尔金', '托爾金', '約翰·羅納德·瑞爾·托爾金', '約翰·羅納德·瑞爾·託爾金', '约翰·罗纳德·瑞尔·托尔金']]",
 "[['溫紐特影片公司'], ['新線影業']]",
 "[['彼得·杰克逊', 'Peter Jackson', '彼得·傑克森', '彼得·積遜', '彼德·積遜'], ['法蘭·華許', '法蘭·沃許', '法蘭·華爾絲'], ['巴利·奧斯朋'], ['索尔·扎恩兹']]"]
```

<b>Example 2:</b>
```python
rep = requests.post(
    url = "http://localhost:8855/kbqa",
    data = {
        "question": "海曙区在哪个城市？"
    }
)
json.loads(rep.content.decode())["output"][:3]
```
this will output
```json
["[['宁波市', '宁波', '甬'], ['海曙区', '海曙']]",
 "['Point(121.39475 29.85648)', 'Point(121.41092 29.78336)', 'Point(121.53333333333 29.883333333333)']",
 "[['高桥镇'], ['鄞江镇'], ['章水镇'], ['古林镇', '古林'], ['横街镇'], ['江厦街道', '江厦街道办事处'], ['望春街道', '望春街道办事处'], ['段塘街道', '段塘街道办事处'], ['洞桥镇'], ['集士港镇'], ['月湖街道', '月湖街道 (宁波市)', '月湖街道办事处'], ['鼓楼街道 (宁波市)'], ['南门街道 (宁波市)'], ['西门街道'], ['白云街道'], ['石碶街道'], ['龙观乡']]"]
```

<b>Example 3:</b>
```python
rep = requests.post(
    url = "http://localhost:8855/kbqa",
    data = {
        "question": "洪都拉斯什么时候的失业率为4.0？"
    }
)
json.loads(rep.content.decode())["output"][:3]
```
this will output
```json
["['2014-01-01T00:00:00Z', 'http://www.wikidata.org/value/c91277cf69500270615dc91eeba92a40']"]
```

<br/>
<h3>
<b>
Recommend you to read below two parts:
</b>
</h3>

<h4>
<p>
<a href="design_construction.md"> Design Construction </a>
</p>
</h4>
This will give you a project summary.

<h4>
<p>
<a href="api_doc.md"> API Documentation </a>
</p>
</h4>
This will help you have a knowledge of the detail function definition.

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/DeepPavlov-Chinese-KBQA](https://github.com/svjack/DeepPavlov-Chinese-KBQA)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
-->
* [DeepPavlov](https://github.com/deeppavlov/DeepPavlov)
* [LC-QuAD-augmentation-toolkit](https://github.com/svjack/LC-QuAD-augmentation-toolkit)
* [Haystack](https://github.com/deepset-ai/haystack)
* [EasyNMT](https://github.com/UKPLab/EasyNMT)
* [adapter-transformers](https://github.com/adapter-hub/adapter-transformers)
* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
* [rdfhdt](https://www.rdfhdt.org/datasets/)
* [rdflib](https://github.com/RDFLib/rdflib)
* [tableQA-Chinese](https://github.com/svjack/tableQA-Chinese)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
