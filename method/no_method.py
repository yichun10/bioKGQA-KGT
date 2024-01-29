import transformers
import torch
from tqdm import tqdm
import json
from transformers import AutoModel, AutoTokenizer, pipeline
from py2neo import Graph
from scipy.spatial.distance import cosine
link1 = Graph("address", auth=("neo4j", "key"))# KG information
model1 = "" #model route

tokenizer = AutoTokenizer.from_pretrained(model1)
pipeline1 = transformers.pipeline(
    "text-generation",
    model=model1,
    torch_dtype=torch.float16,
    device_map="auto", 
)
dataset = []

def exception_(old_question):
    out=""
    prompt = """
    Answer the question . 
    Note: The output format is: Output: One sentence in natural language.
    """
    input_text1 = prompt +old_question
    sequences = pipeline1(
        input_text1,
        do_sample=True,
        top_k=10,
        return_full_text=False,
        top_p = 0.7,  
        temperature = 0.8, 
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200, 
    )
    for seq in sequences:
        lines = seq['generated_text'].split('\n')
        for line in lines:
            if line.strip():  
                out=line
                break  
    dataset.append(
        {
        "question":old_question,
        "answer": out
        }
    )
file_path = ''  # Replace with your test JSON file path
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
for item in tqdm(data):
    try:
        old_question = item['question']

        #text2cypher
        prompt = """
        Given a knowledge graph, answer the question according to the schema of the knowledge graph.
        The schema of the knowledge graph includes the following:
        (Drug)-[:activation_to {}]->(Genesymbol),(Drug)-[:treatment {}]->(Cancer)...
        The attribute list is:
        "Drug": "drug.id, drug.name...
        """
        question_pro="""
        Create a Cypher statement to answer the following question:
        """
        input_text1 = prompt + question_pro +old_question
        sequences = pipeline1(
            input_text1,
            do_sample=True,
            top_k=10,
            return_full_text=False,
            top_p = 0.95,  
            temperature = 0.01,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=3000, 
        )
        for seq in sequences:
            lines = seq['generated_text'].split('\n')
            for line in lines:
                if line.strip(): 
                    break  

        cypher_query=line
        answers = link1.run(cypher_query)
        knowledge_prompts = ""
        for ans in answers:
            if len(ans) > 1:
                knowledge_prompts += f"{ans[0]} {ans[1]}\n"
            else:
                knowledge_prompts += f"{ans[0]}\n"

        lines = knowledge_prompts.split('\n')[:10]
        result_string_with_separator = '\n'.join(lines)


        #inference
        prompt = """
        Answer my question based on the relationship chain. If there is no relationship chain, answer my question based on your existing knowledge.
        Note: The output format is: Output: One sentence in natural language.
        """

        input_text1 =prompt+old_question+result_string_with_separator
        sequences = pipeline1(
            input_text1,
            do_sample=True,
            top_k=10,
            return_full_text=False,
            top_p = 0.95,  
            temperature = 0.1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1000, 
        )
        for seq in sequences:
            lines = seq['generated_text'].split('\n')
            for line in lines:
                if line.strip():  
                    out=line
                    break  

        dataset.append(
            {
            "question":old_question,
            "answer": out
            }
        )
        
    except Exception as e:
        exception_(old_question)
json.dump(dataset, open('output_route', 'w', encoding='utf-8'), indent=4,ensure_ascii=False)