def Subgraph(old_question, link1, pipeline1, tokenizer,first_pro,entity_pro,name_pro,que_pro,attr_pro):
    result_string_with_separator=""
    prompt = """
    Given a knowledge graph, answer the question according to the schema of the knowledge graph.
    example :
    Create a Cypher statement to answer the following question:What targeted drugs do A have?The name of the head entity is a.No intermediate entity.The relationship chain pattern is (Drug)-[:inhibition_to {}]->(Cancer).The selected attribute is durg.class_type.
    MATCH p=(drug:Drug)-[:inhibition_to]-(cancer:Cancer) WHERE cancer.name='a' RETURN p,durg.class_type limit 10
    Create a Cypher statement to answer the following question:What drug is a resistant to a?The name of the head entity is a.The head entity type is SnvFull,the middle entity type is CancerCell, and the tail entity type is Drug.The relationship chain pattern is (CancerCell)-[:has_var {}]->(SnvFull), (CancerCell)-[:resistance_to {}]->(Drug).The selected attribute is drug.name.
    MATCH p=(snvfull:SnvFull)-[:has_var]-(cancercell:CancerCell)-[:resistance_to]-(drug:Drug) WHERE snvfull.name='a' RETURN p,drug.name limit 10
    Create a Cypher statement to answer the following question:What are the common snvfull in ovarian cancer?The name of the head entity is 卵巢癌.The head entity type is Cancer,the middle entity type is CancerCell, and the tail entity type is SnvFull.The relationship chain pattern is (CancerCell)-[:originated_from {}]->(Cancer),(CancerCell)-[:has_var {}]->(SnvFull).The selected attribute is snvfull.name.
    MATCH p=(cancer:Cancer)-[:originated_from]-(cancercell:CancerCell)-[:has_var]-(snvfull:SnvFull) WHERE cancer.name = "卵巢癌" RETURN p,snvfull.name limit 10
    Create a Cypher statement to answer the following question:What targeted drugs for a?No intermediate entity.The specific name of the head entity is:b,c,d。The relationship chain pattern is (Drug)-[:treatment {}]->(Cancer).The selected attribute is drug.name.
    MATCH p=(drug:Drug)-[:treatment]-(cancer:Cancer) WHERE cancer.name IN ['b', 'c', 'd'] RETURN p,drug.name limit 10
    Create a Cypher statement to answer the following question:What is the relationship between AFF1 and the skin melanoma?Relationship detected.The name of the head entity is AFF1 and The name of the tail entity is 皮肤黑色素瘤.
    MATCH p= shortestPath((a)-[*..3]-(b)) WHERE a.name = "AFF1" AND b.name = "皮肤黑色素瘤" RETURN p
    Create a Cypher statement to answer the following question:What kind of drug is aletinib?The name of the head entity is 阿来替尼.Detected attribute.The entity type is Drug.
    MATCH (n:Drug{name:'阿来替尼'})  RETURN n.name,n.description
    Create a Cypher statement to answer the following question:What snvfull do piperacillin need to be tested for?The name of the head entity is 哌柏西利.The head entity type is Drug,the middle entity type is CancerCell, and the tail entity type is SnvFull.The relationship chain pattern is (CancerCell)-[:has_var {}]->(SnvFull), (CancerCell)-[:resistance_to {}]->(Drug).The selected attribute is snvfull.exact_chgvs.
    MATCH p=(drug:Drug)-[:inhibition_to]-(genesymbol:Genesymbol)-[:has_gene]-(snvfull:SnvFull) WHERE drug.name = "哌柏西利" RETURN p,snvfull.name limit 10
    """

    question_pro="""
    Create a Cypher statement to answer the following question:
    """
    input_text1 = prompt + question_pro +old_question+first_pro+entity_pro+name_pro+que_pro+attr_pro
    sequences = pipeline1(
        input_text1,
        do_sample=True,
        top_k=10,
        return_full_text=False,
        top_p = 0.95,  #0.9
        temperature = 0.01, #0.2
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
    return cypher_query,result_string_with_separator