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
            ("CancerCell", "Drug", {"relation": "resistance_to"}),  
            ("Drug", "CancerCell", {"relation": "resistance_to"}), 
            ("CancerCell", "Cancer", {"relation": "originated_from"}),  
            ("Cancer", "CancerCell", {"relation": "originated_from"}), 
            ("Fusion", "Genesymbol", {"relation": "has_3gene"}),
            ("SnvFull", "Genesymbol", {"relation": "has_gene"}),
            ("Genesymbol", "SnvFull", {"relation": "has_gene"}),
            ("Genesymbol", "Genesymbol", {"relation": "positive_regulated"}),
            ("Fusion", "Genesymbol", {"relation": "has_5gene"}),
            ("Genesymbol", "Genesymbol", {"relation": "negative_regulated"}),
            ("Genesymbol", "GeneticDisease", {"relation": "cause_to"}),
            ("ClinicalTrial", "Drug", {"relation": "include_a"}),
            ("Pathway", "Genesymbol", {"relation": "include_a"}),
            ("Genesymbol", "Pathway", {"relation": "include_a"}),
            ("ClinicalTrial", "Cancer", {"relation": "include_a"}),
            ("Cancer", "ClinicalTrial", {"relation": "include_a"}),
            ("ClinicalTrial", "Genesymbol", {"relation": "include_a"}),
            ("Genesymbol", "ClinicalTrial", {"relation": "include_a"}),
            ("CancerCell", "SnvPartial", {"relation": "has_var"}),
            ("SnvPartial", "CancerCell", {"relation": "has_var"}),
            ("CancerCell", "Expression", {"relation": "has_var"}),
            ("CancerCell", "CompoundMutation", {"relation": "has_var"}),
            ("CancerCell", "Fusion", {"relation": "has_var"}),
            ("CancerCell", "CNA", {"relation": "has_var"}),
            ("CancerCell", "SnvDrugrule", {"relation": "has_var"}),
            ("SnvDrugrule", "CancerCell", {"relation": "has_var"}),
            ("CancerCell", "SnvFull", {"relation": "has_var"}),
            ("SnvFull", "CancerCell", {"relation": "has_var"}),
            ("Drug", "Genesymbol", {"relation": "inhibition_to"}),
            ("CancerCell", "Drug", {"relation": "sensitivity_to"}),
            ("Drug", "CancerCell", {"relation": "sensitivity_to"}),#
            ("CancerCell", "DrugComb", {"relation": "sensitivity_to"}),
            ("Genesymbol", "Genesymbol", {"relation": "synthetic_lethality"}),
            ("GeneticDisease", "Cancer", {"relation": "develop_to"}),
            ("DrugType", "DrugType", {"relation": "subclass_of"}),
            ("Cancer", "Cancer", {"relation": "subclass_of"}),
            ("CompoundMutation", "SnvFull", {"relation": "has_a"}),
            ("SnvFunction", "SnvFull", {"relation": "has_a"}),
            ("SnvCarcinogenicity", "SnvFull", {"relation": "has_a"}),
            ("SnvCarcinogenicity", "Drug", {"relation": "has_a"}),
            ("SnvPartial", "SnvFull", {"relation": "has_a"}),
            ("DrugComb", "Drug", {"relation": "has_a"}),
            ("SnvPathogenic", "SnvFull", {"relation": "has_a"}),
            ("CompoundMutation", "Drug", {"relation": "has_a"}),
            ("DrugType", "Drug", {"relation": "has_a"}),
            ("DrugType", "SnvFull", {"relation": "has_a"}),
            ("DrugComb", "SnvFull", {"relation": "has_a"}),
            ("SnvDrugrule", "SnvFull", {"relation": "has_a"}),
            ("Genesymbol", "Cancer", {"relation": "driving_to"}),
            ("Cancer", "Genesymbol", {"relation": "driving_to"}),
            ("Gene_Alias", "Soc", {"relation": "is_a"}),
            ("CancerAlias", "Genesymbol", {"relation": "is_a"}),
            ("Gene_Alias", "Genesymbol", {"relation": "is_a"}),
            ("DrugAlias", "Soc", {"relation": "is_a"}),
            ("CancerAlias", "Soc", {"relation": "is_a"}),
            ("Pt", "Cancer", {"relation": "is_a"}),
            ("DrugAlias", "Drug", {"relation": "is_a"}),
            ("CancerAlias", "Drug", {"relation": "is_a"})
        ]  
        G.add_edges_from(edges)  

        node_information = """(Drug)-[:activation_to {}]->(Genesymbol),(Drug)-[:treatment {}]->(Cancer),(CancerCell)-[:resistance_to {}]->(DrugComb),(CancerCell)-[:resistance_to {}]->(Drug),(CancerCell)-[:originated_from {}]->(Cancer),(Fusion)-[:has_3gene {}]->(Genesymbol),(SnvCarcinogenicity)-[:has_gene {}]->(Genesymbol),(Expression)-[:has_gene {}]->(Genesymbol),(SnvPartial)-[:has_gene {}]->(Genesymbol),(CNA)-[:has_gene {}]->(Genesymbol),(SnvDrugrule)-[:has_gene {}]->(Genesymbol),(SnvPathogenic)-[:has_gene {}]->(Genesymbol),(SnvFull)-[:has_gene {}]->(Genesymbol),(SnvFunction)-[:has_gene {}]->(Genesymbol),(Genesymbol)-[:positive_regulated {}]->(Genesymbol),(Fusion)-[:has_5gene {}]->(Genesymbol),(Drug)-[:induce_to {}]->(Pt),(Genesymbol)-[:negative_regulated {}]->(Genesymbol),(Genesymbol)-[:cause_to {}]->(GeneticDisease),(Pathway)-[:include_a {}]->(Cancer),(ClinicalTrial)-[:include_a {}]->(Drug),(Pathway)-[:include_a {}]->(Drug),(Pathway)-[:include_a {}]->(Genesymbol),(ClinicalTrial)-[:include_a {}]->(Cancer),(ClinicalTrial)-[:include_a {}]->(Genesymbol),(CancerCell)-[:has_var {}]->(SnvPartial),(CancerCell)-[:has_var {}]->(Expression),(CancerCell)-[:has_var {}]->(CompoundMutation),(CancerCell)-[:has_var {}]->(Fusion),(CancerCell)-[:has_var {}]->(CNA),(CancerCell)-[:has_var {}]->(SnvDrugrule),(CancerCell)-[:has_var {}]->(SnvFull),(Drug)-[:inhibition_to {}]->(Genesymbol),(CancerCell)-[:sensitivity_to {}]->(Drug),(CancerCell)-[:sensitivity_to {}]->(DrugComb),(Genesymbol)-[:synthetic_lethality {}]->(Genesymbol),(GeneticDisease)-[:develop_to {}]->(Cancer),(DrugType)-[:subclass_of {}]->(Cancer),(DrugType)-[:subclass_of {}]->(DrugType),(Cancer)-[:subclass_of {}]->(DrugType),(Cancer)-[:subclass_of {}]->(Cancer),(CompoundMutation)-[:has_a {}]->(SnvFull),(SnvFunction)-[:has_a {}]->(SnvFull),(SnvCarcinogenicity)-[:has_a {}]->(SnvFull),(SnvCarcinogenicity)-[:has_a {}]->(Drug),(SnvDrugrule)-[:has_a {}]->(Drug),(SnvPartial)-[:has_a {}]->(SnvFull),(DrugComb)-[:has_a {}]->(Drug),(SnvPathogenic)-[:has_a {}]->(SnvFull),(SnvPathogenic)-[:has_a {}]->(Drug),(SnvFunction)-[:has_a {}]->(Drug),(SnvPartial)-[:has_a {}]->(Drug),(CompoundMutation)-[:has_a {}]->(Drug),(DrugType)-[:has_a {}]->(Drug),(DrugType)-[:has_a {}]->(SnvFull),(DrugComb)-[:has_a {}]->(SnvFull),(SnvDrugrule)-[:has_a {}]->(SnvFull),(Genesymbol)-[:driving_to {}]->(Cancer),(Gene_Alias)-[:is_a {}]->(Soc),(Gene_Alias)-[:is_a {}]->(Cancer),(CancerAlias)-[:is_a {}]->(Genesymbol),(Gene_Alias)-[:is_a {}]->(Genesymbol),(DrugAlias)-[:is_a {}]->(Soc),(CancerAlias)-[:is_a {}]->(Soc),(Gene_Alias)-[:is_a {}]->(Drug),(Pt)-[:is_a {}]->(Cancer),(DrugAlias)-[:is_a {}]->(Drug),(CancerAlias)-[:is_a {}]->(Drug),(DrugAlias)-[:is_a {}]->(Genesymbol)
        """
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
