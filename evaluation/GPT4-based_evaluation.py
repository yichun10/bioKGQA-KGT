import json
import requests
import time
from tqdm import tqdm

counter = 0 
def query_chatgpt(prompt, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [{"role": "system", "content": "You are a rating robot."}, 
                     {"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

api_key = ""  # your API key
similarities = []
with open('/your_path/test.json', 'r') as file:
    test_datas = json.load(file)

with open('/your_path/QA.json', 'r') as file:
    data = json.load(file)

for record,ceping_data in tqdm(zip(test_datas,data)):
    test_answer = record['answer']
    answer = ceping_data['answer']
    try:
        prompt = f"""
        I have two sentences:{answer} and {test_answer} You need to rate their similarity, and if they express exactly the same meaning, give them 100 points; If the expressed meaning is opposite, give 0 points; If there are any unreasonable aspects, corresponding points will be deducted and specific scores will be given.
        Note: Only the numerical part of the score needs to be output.
        For examples: 50
        """  
        response = query_chatgpt(prompt, api_key)
        sentence=response['choices'][0]['message']['content']
        value=int(sentence)
        similarities.append(value)

        # Pause for one second after each call
        counter += 1 
        if counter % 1 == 0:
            time.sleep(1)
    except Exception as e:
        print(e)
        similarities.append(0)
average_similarity = sum(similarities) / len(similarities)
print("GPT4 Average Similarity:", average_similarity)
