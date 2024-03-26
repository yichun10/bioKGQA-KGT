import networkx as nx
import torch  
def schema_inf(old_question, first_element, second_element, model1, tokenizer, calculate_similarity,target1,attr_pro,name_pro,name_ht):    
    if not target1:
        gxl_pro=""
        G = nx.DiGraph() 
        edges = [  
            ("Drug", "Genesymbol", {"relation": "activation_to"}), 
            ("Genesymbol", "Drug", {"relation": "activation_to"}),
            ("Drug", "Cancer", {"relation": "treatment"}),  
            ("Cancer", "Drug", {"relation": "treatment"}), 
            ("CancerCell", "DrugComb", {"relation": "resistance_to"}),  
            ("CancerCell", "Drug", {"relation": "resistance_to"}),  
            #......
        ]  #Due to copyright issues, the ellipsis portion is temporarily unavailable for publication. You can add it according to the format after receiving the KG.
        G.add_edges_from(edges)  

        node_information = """(Drug)-[:activation_to {}]->(Genesymbol),(Drug)-[:treatment {}]->(Cancer)......
        """#Due to copyright issues, the ellipsis portion is temporarily unavailable for publication. You can add it according to the format after receiving the KG.
        shortest_paths = nx.all_shortest_paths(G, source=first_element, target=second_element)
        path_prompts = ""
        rel_links = [] 
        mid_eles = []
        for index, path in enumerate(shortest_paths):
            mid_ele = path[1]
            path_prompts += str(path[1])  

            rel_prompts_1 = []
            rel_prompts_2 = []
            mid_elements = []
            entity_pro=""
            links = node_information.strip('\n').split(',')
            for link in links:
                head = link.split('-')[0].strip('(').strip(')')
                tail = link.split('->')[1].strip('(').strip(')')
                if second_element == mid_ele:
                    entity_pro="No intermediate entity."
                    if head == first_element and tail == mid_ele:
                        rel_prompts_1.append(str(link))
                        mid_elements.append(mid_ele)
                    if head == mid_ele and tail == first_element:
                        rel_prompts_1.append(str(link))
                        mid_elements.append(mid_ele)
                else:
                    
                    if head == first_element and tail == mid_ele:
                        rel_prompts_1.append(str(link))
                        mid_elements.append(mid_ele)
                    if head == mid_ele and tail == first_element:
                        rel_prompts_1.append(str(link))
                        mid_elements.append(mid_ele)
                    if head == second_element and tail == mid_ele:
                        rel_prompts_2.append(str(link))
                    if head == mid_ele and tail == second_element:
                        rel_prompts_2.append(str(link))

            for r_i, rel_1 in enumerate(rel_prompts_1):
                mid_eles.append(mid_elements[r_i])
                if rel_prompts_2 == []:
                    rel_links.append([rel_1])
                else:
                    for rel_2 in rel_prompts_2:
                        rel_links.append([rel_1, rel_2])

        similarity_scores = []
        sentence1 = old_question

        for rel_link in rel_links:
            sentence2 = ','.join(rel_link)
            similarity = calculate_similarity(sentence1, sentence2, model1, tokenizer)
            similarity_scores.append(similarity)

        selected_index = torch.tensor(similarity_scores).topk(1)[1][0]
        gxl_pro = ','.join(rel_links[selected_index])
        if len(rel_links[selected_index])>1:
            mid_ele = mid_eles[selected_index]
            entity_pro=f"The head entity type is {first_element}, the middle entity type is {mid_ele}, and the tail entity type is {second_element}."
        else:
            entity_pro="No intermediate entity."
        que_pro=f"The relationship chain pattern is {gxl_pro}."

    if target1=="Relationship detected":
        entity_pro="Relationship detected."
        name_pro=""
        que_pro=""
        first_pro=""
        attr_pro=f"{name_ht}."
    if target1=="Detected attribute":
        entity_pro="Detected attribute."
        name_pro=f"The entity type is {second_element}."
        que_pro=""
        attr_pro=""
    return entity_pro, que_pro, attr_pro, name_pro