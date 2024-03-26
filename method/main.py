import torch
import transformers
from tqdm import tqdm
import json
from transformers import AutoModel, AutoTokenizer, pipeline
from user_input import link1, file_path, model1
from functions import get_sentence_embedding, calculate_similarity, exception_
from Question_analysis import QA_Q
from Graph_Schema_based_inference import schema_inf
from Subgraph_construction import Subgraph
from Inference import infer

dataset = []
"""
{
    "question":old_question,
    "answer": out
}
"""
tokenizer = AutoTokenizer.from_pretrained(model1)
pipeline1 = transformers.pipeline(
    "text-generation",
    model=model1,
    torch_dtype=torch.float16,
    device_map="auto", 
)

model1 = AutoModel.from_pretrained(model1)
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
for item in tqdm(data):
    try:
        old_question = item['question']
        first_pro, first_element, second_element, name_pro, attr_pro, target1,name_ht=QA_Q(old_question, link1, pipeline1, tokenizer)
        entity_pro, que_pro, attr_pro, name_pro=schema_inf(old_question, first_element, second_element, model1, tokenizer, calculate_similarity,target1,attr_pro,name_pro,name_ht)
        cypher_query,result_string_with_separator=Subgraph(old_question, link1, pipeline1, tokenizer,first_pro,entity_pro,name_pro,que_pro,attr_pro)
        out=infer(old_question, pipeline1, tokenizer,result_string_with_separator)
        dataset.append(
            {
            "question":old_question,
            "answer": out
            }
        )
        json.dump(dataset, open('/answer/codellama13_SOKG.json', 'w', encoding='utf-8'), indent=4,ensure_ascii=False)
    except Exception as e:
        print(e)
        out=exception_(old_question,pipeline1,tokenizer)
        dataset.append(
            {
            "question":old_question,
            "answer": out
            }
        )
        json.dump(dataset, open('/answer/codellama13_SOKG.json', 'w', encoding='utf-8'), indent=4,ensure_ascii=False)