from rouge import Rouge 
import json
rough_recall=[]
rough_precision=[]
rough_f1=[]
def calculate_rouge_score(sentence1, sentence2):
    rouge = Rouge()
    scores = rouge.get_scores(sentence1, sentence2)
    return scores[0]['rouge-l']
with open('/your_path/test.json', 'r') as file:
    test_datas = json.load(file)

with open('/your_path/QA.json', 'r') as file:
    data = json.load(file)
similarities = []
for record,ceping_data in zip(test_datas,data):
    sentence1 = ceping_data['answer']
    sentence2 = record['answer']
    rouge_score = calculate_rouge_score(sentence1, sentence2)
    rough_recall.append(rouge_score['r'])
    rough_precision.append(rouge_score['p'])
    rough_f1.append(rouge_score['f'])

average_rough_recall = sum(rough_recall) / len(rough_recall)
average_rough_precision = sum(rough_precision) / len(rough_precision)
average_rough_f1 = sum(rough_f1) / len(rough_f1)

print("average_rough_recall:", average_rough_recall)
print("average_rough_precision:", average_rough_precision)
print("average_rough_f1:", average_rough_f1)