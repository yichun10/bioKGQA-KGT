# Knowledge Graph-based Thought: a knowledge graph enhanced LLMs framework for pan-cancer question answering
## Introduction
In recent years, Large Language Models (LLMs) have made significant progress in various fields, particularly garnering widespread attention in biomedical sciences. Despite their extraordinary capabilities, the application of LLMs in real-world scenarios is still hampered by issues such as erroneous outputs and hallucinatory responses. To this end, we introduce the Knowledge Graph-based Thought (KGT) framework. This innovative architecture seamlessly integrates LLMs with Knowledge Graphs (KGs). The primary function of KGT is to use verifiable information from KGs to optimize the initial responses of LLMs. This process effectively reduces factual errors during reasoning. And the KGT framework possesses strong adaptability and is easily integrated with various LLMs. It performs commendably with open-source LLMs and is capable of adapting to different biomedical knowledge graphs. The application of the KGT framework to pan-cancer KGs significantly enhances the potential of LLMs. KGT can facilitate the discovery of new uses for existing drugs through potential drug-cancer associations, and can assist in predicting resistance by analyzing relevant biomarkers and genetic mechanisms. Furthermore, we employ a pan-cancer knowledge graph, called SmartQuerier Oncology Knowledge Graph, and develop a biomedical question-answering dataset based on the knowledge graph, aiming at evaluating the effectiveness of our framework. The experimental results showcase the exceptional performance of KGT in the field of biomedical question answering.

## Framework of KGT
<p align="center">
<img width="925" alt="image" src="https://github.com/yichun10/bioKGQA/assets/156771528/1906da6c-710d-4974-94e5-445c05f1cf88">
</p>
Fig. 2. Framework of KGT. (A) Question analysis. Decompose the question and extract its key information. (B) Graph Schema-based inference. Input
the types of the head and tail entities into the graph schema of the knowledge graph, complete the graph reasoning, and obtain the optimal relational
chain pattern. (C) Subgraph construction. Generate a query statement and retrieve the subgraph. (D) Inference. Complete the final reasoning and
output the results in natural language.

## Demo
You can see our demo through the following video.
Due to limited resources, we do not currently provide online demos. If you need an online demo, you can send an email to fengyichun22@mails.ucas.ac.cn. We suggest downloading the model for local testing.

## Environments Setting
1、In a conda env with PyTorch / CUDA available clone and download this repository.
2、In the top-level directory run:
```bash
pip install
```
## Installation
* CUDA>=11.3, Python>=3.8
* GPU Requirements: Two V100 cards

## Usage
### Data preparation
You can obtain the SOKG_dataset.json from the dataset folder, which is a dataset containing 405 question-answer pairs.For access to the SmartQuerier Oncology Knowledge Graph, please contact at service@smartquerier.com.

### Downloading the Required LLMs
You can download the corresponding model from the official website. For instance, if you need to download CodeLlama, you can visit the link [](https://github.com/facebookresearch/codellama.git)https://github.com/facebookresearch/codellama.git.
### Test
You can obtain the KGT.py from the method folder.
```bash
link1 = Graph("address", auth=("neo4j", "key"))
model1 = ""
file_path = '' 
```
Enter the Knowledge Graph API, download paths for the LLM and dataset, and you can directly run the KGT.py.
### Evaluation
We have designed three evaluation methods: ROUGE, BERT score, and an evaluator based on GPT-4. You simply need to enter the paths for the test set and the path for the generated answers in the corresponding fields to run it directly.
```bash
with open('/your_path/test.json', 'r') as file:
    test_datas = json.load(file)

with open('/your_path/QA.json', 'r') as file:
    data = json.load(file)

```
