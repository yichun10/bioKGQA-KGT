from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
import networkx as nx  
from transformers import AutoTokenizer
import transformers
import torch
import openai
from tqdm import tqdm
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from py2neo import Node, Graph, Relationship,NodeMatcher
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging
from utils import load_model_on_gpus
logger = logging.getLogger('gradio_log')
logger.setLevel(logging.DEBUG)  
file_handler = logging.FileHandler('gradio_log.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

link1 = Graph("address", auth=("neo4j", "password"))# KG information.
model1 = "./model/codellama/CodeLlama-13b-Instruct/13b_hf" #Your model path
tokenizer = AutoTokenizer.from_pretrained(model1)
pipeline1 = transformers.pipeline(
    "text-generation",
    model=model1,
    torch_dtype=torch.float16,
    device_map="auto", 
)

"""Override Chatbot.postprocess"""
dataset = []
"""
{

    "answer": out,
    "answer_list": binyu,
}
"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):

        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def process_question(input, chatbot, max_length, top_p, temperature,history, past_key_values):
    chatbot.append((parse_text(input), ""))
    old_question = input
    logger.info("Function started with input: " + str(input))
    # Your code here,for example:
    prompt = """
    You are a reasoning robot, and you need to output natural language to answer my questions.
    """
    input_text1 = prompt +old_question
    sequences = pipeline1(
        input_text1,
        do_sample=True,
        top_k=10,
        return_full_text=False,
        top_p = 0.7,  #0.9
        temperature = 0.8, #0.2
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1000, 
    )

    for seq in sequences:
        lines = seq['generated_text'].split('\n')
        for line in lines:
            if line.strip():  
                out_put=line
                break  

    dataset.append(
        {
        "question":old_question,
        "cypher":[],
        "answer": out_put
        }
    )
    logger.error("Error occurred: " + str(e))
    logger.info("Function completed with out_put: " + str(out_put))
    json.dump(dataset, open('./answer/web_log.json', 'w', encoding='utf-8'), indent=4,ensure_ascii=False)
    history=out_put
    past_key_values=past_key_values
    chatbot[-1] = (parse_text(input), parse_text(out_put))
    return chatbot,history, past_key_values

def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">bioKGQA-KGT</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(process_question, [user_input, chatbot, max_length, top_p, temperature,history, past_key_values],
                    [chatbot,history, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[chatbot,history, past_key_values], show_progress=True)

demo.queue().launch(share=True, inbrowser=True)