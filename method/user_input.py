from py2neo import Graph
link1 = Graph("address", auth=("neo4j", "password"))  # KG information.
file_path = './dataset/SOKG.json'  # Replace with your test JSON file path
model1 = "./model/codellama/CodeLlama-13b-Instruct/13b_hf"  # Your model path
