<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">DeepPavlov-Chinese-KBQA</h3>

  <p align="center">
   		基于DeepPavlov的中文知识问答系统
    <br />
  </p>
</p>

[In English](README_EN.md)

### 简要引述

[DeepPavlov](https://github.com/deeppavlov/DeepPavlov) 是一个基于TensorFlow、Keras和PyTorch构建的开源对话AI库，它具有知识库问答模型，支持对英文和俄文进行知识库问答。

本项目旨在通过将相应的其它语言语义解析组件替换为中文版本，并在中文领域进行微调，重构一个基于中文的KBQA系统。

### 安装
请参考[INSTALL.sh](INSTALL.sh)来安装环境，并确保您能够运行原始DeepPavlov项目的KBQA任务。<br/>

由于安装可能会比较困难，我已经建立了一个包含预训练模型的Docker镜像，这些模型位于model、sel_ner、tmp_cls和ranker_cls文件夹中。如果有需要，您可以发送邮件至ehangzhou@outlook.com来获取它们。<br/>

Wikidata知识库的hdt文件可以从我这里获取，或者从[rdfhdt](https://www.rdfhdt.org/datasets/)下载，并确保您下载的版本包含中文部分（有些只有英文部分）。<br/>

预训练模型应放置在相应的位置。<br/>

<!--
files below kbqa-explore path can be download following [installtion of LC-QuAD-augmentation-toolkit](https://github.com/svjack/LC-QuAD-augmentation-toolkit)'s [Baidu Yun Drive link](https://pan.baidu.com/s/1e66Lt6nisM3583dbIGsO5w?pwd=ntwz) ,others can email me to get.
-->
您可以使用以下链接从百度云盘获取它们，并将它们放置在项目根路径中。
<br/>
https://pan.baidu.com/s/1HLAzfBPasudGqtp9f0j5iA?pwd=p83h
<br/>
请记得使用cat将wikidata.hdt.aa、wikidata.hdt.ab和wikidata.hdt.ac合并为wikidata.hdt后再使用它。

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

### 知识图谱问答使用方式
您可以直接运行kbqa_step.py（在main闭包中调用do_search），或者进入mysite并激活您安装KBQA的环境（在INSTALL.sh中）。<br/>
```bash
conda activate kbqa_env
bash run.sh
```
这将在8855端口运行一个服务，然后您可以使用它来检索类似于结论的内容。<br/>
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
推荐你阅读下面的两部分:
</b>
</h3>

<h4>
<p>
<a href="design_construction.md"> 设计结构（英文） </a>
</p>
</h4>
这个将会梳理你对于工程的整体脉络

<h4>
<p>
<a href="api_doc.md"> API 文档（英文） </a>
</p>
</h4>
这部分将会让你对于细节函数设计有了解

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
