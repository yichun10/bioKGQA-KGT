from transformers import BertTokenizer, BertModel
import torch
import json

with open('/your_path/test.json', 'r') as file:
    test_datas = json.load(file)

with open('/your_path/QA.json', 'r') as file:
    data = json.load(file)
similarities = []
for record,ceping_data in zip(test_datas,data):
    try:
        sentence1 = ceping_data['answer']
        sentence2 = record['answer']
        model_name = '/path/bert'  # Select the corresponding pre trained model
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        inputs = tokenizer([sentence1, sentence2], return_tensors='pt', padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state

        sentence1_embedding = embeddings[0, 0, :]  
        sentence2_embedding = embeddings[1, 0, :]  

        cos = torch.nn.CosineSimilarity(dim=0)
        similarity = cos(sentence1_embedding, sentence2_embedding)
        similarities.append(similarity)

    except Exception as e:
        similarity = 0
        similarities.append(similarity)
average_similarity = sum(similarities) / len(similarities)
print("Bertscores Average Similarity:", average_similarity)