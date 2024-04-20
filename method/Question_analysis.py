import re
def QA_Q(old_question, link1, pipeline1, tokenizer):
    name_ht,first_name,first_pro, first_element, second_element, name_pro, attr_pro,third_element,target1 = "", "", "", "", "","","","",""
    prompt = """
    You need to follow these three steps based on your biomedical knowledge:
    step1:Extract the name of the head entity and the type of the tail entity in my question,with the head defined as the active voice of the problem and the tail defined as the passive voice of the problem. The entity types include: Drug, Genesymbol, Cancer, CancerCell, Fusion, SnvCarcinogenicity, Expression, SnvPartial, CNA, SnvDrugrule, SnvPathogenic, SnvFull, SnvFunction, GeneticDisease, Pathway, ClinicalTrial, Soc, CancerAlias, DrugAlias. 
    step2:Based on the entity type from step 1, select an attribute from the attribute list that best fits my question.
    step3:If there is only one head entity name, the output format should be(head entity name, tail entity type, attribute); if there are two head entity names, the output format should be(head entity name 1,tail entity type, attribute),(head entity name 2,tail entity type, attribute).

    The attribute list is:
    "Drug": "drug.id, drug.name, drug.name_en, drug.description, drug.class_type, drug.nmpa_approved, drug.fda_approved, drug.commodity_name",
    "Cancer":"cancer.name, cancer.description, cancer.id, cancer.name_en",
    "Genesymbol":"genesymbol.id, genesymbol.name, genesymbol.grch37_refseq, genesymbol.tsg, genesymbol.description, genesymbol.oncogene, genesymbol.genesymbol_ncbi,genesymbol.full_name",
    "GeneticDisease":"geneticdisease.id, geneticdisease.name, geneticdisease.name_en, geneticdisease.description,geneticdisease.omim_id",
    "ClinicalTrial":"clinicaltrial.id, clinicaltrial.name, clinicaltrial.description, clinicaltrial.url, clinicaltrial.start_phase, clinicaltrial.end_phase, clinicaltrial.target_size, clinicaltrial.recruitment_time, clinicaltrial.recruitment_status, clinicaltrial.countries, clinicaltrial.min_age, clinicaltrial.max_age, clinicaltrial.gender",
    "CancerCell":"cancercell.id, cancercell.name",
    "CNA":"cna.id, cna.name, cna.cna_val",
    "Expression":"expression.id, expression.name, expression.exp_value",
    "Fusion":"fusion.id, fusion.name",
    "SnvDrugrule":"snvdrugrule.id, snvdrugrule.name, snvdrugrule.drug_rule",
    "SnvPartial":"snvpartial.id, snvpartial.name, snvpartial.partial_pos, snvpartial.variant_type",
    "SnvPathogenic":"snvpathogenic.id, snvpathogenic.name, snvpathogenic.pathogenic",
    "SnvFull":"snvfull.id, snvfull.name, snvfull.biological_effect, snvfull.clinvar_significant, snvfull.exact_chgvs, snvfull.exact_phgvs, snvfull.oncogenic, snvfull.variant_type",
    "SnvFunction":"snvfunction.id, snvfunction.name",
    "SnvCarcinogenicity":"snvcarcinogenicity.id, snvcarcinogenicity.name",
    "Pathway":"pathway.id, pathway.description, pathway.keggpathway_id, pathway.name",
    "CompoundMutation":"compoundmutation.id, compoundmutation.name",
    "DrugType":"drugtype.id, drugtype.atc_code, drugtype.name, drugtype.drugtype_name_en",
    "Pt":"pt.id, pt.pt_code, pt.name, pt.pt_name_en",
    "Soc":"soc.id, soc.soc_code, soc.name, soc.soc_name_en",
    "CancerAlias":"canceralias.id, canceralias.name",
    "DrugAlias":"drugalias.id, drugalias.name",
    
    For example:
    What type of cancer can bexarotene treat?(bexarotene,Cancer,cancer.name)
    What are the targeted drugs for ERBB2 in lung cancer?(lung cancer,Drug,drug.class_type),(ERBB2,Drug,drug.class_type)
    """
    input_text1 = prompt + old_question
    sequences = pipeline1(
        input_text1,
        do_sample=True,
        top_k=10,
        return_full_text=False,
        top_p = 0.95,  
        temperature = 0.01,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2000, 
    )
    for seq in sequences:
        lines = seq['generated_text'].split('\n')
        for line in lines:
            if line.strip(): 
                # print(line)  
                break  

    count = line.count('(')
    if 'relation' in old_question:
        target1 = "Relationship detected"
    if count == 1:
        first_element = line.split(',')[0].strip('() ')
        second_cypher_query=f"""
        MATCH (e) 
        WHERE e.name_en = '{first_element}'
        RETURN e.name
        """
        name_ans=link1.run(second_cypher_query)
        for ans in name_ans:
            if len(ans)==1:
                first_element = str(ans[0])  
        first_pro=f"The name of the head entity is {first_element}."
        cypher_query=f"""
        MATCH (n {{name: '{first_element}'}}) 
        RETURN labels(n)
        """
        answers = link1.run(cypher_query)
        entity_answers = ""
        for ans in answers:
            entity_answers += str(ans[0])  
        match = re.search(r"\['([^]]+)'\]", entity_answers)
        extracted_str = match.group(1) 
        second_element = line.split(',')[1].strip(')').strip()
        
        third_element=line.split(',')[2].strip(')').strip()
        com_question=f"The entity type of {first_element} is {extracted_str}."
        question=old_question+com_question
        first_element = extracted_str 
        answers = link1.run(cypher_query)
        for ans in answers:
            if isinstance(ans[0], list) and ans[0][0] == second_element:
                target1="Detected attribute"
    elif count == 2:
        second_element=""
        matches = re.findall(r'\(([^,]+),', line)
        pattern = r"\(([^,]+),([^,]+),([^)]+)\)"
        matches2 = re.search(pattern, line)
        second_element = matches2.group(2)
        third_element = matches2.group(3)
        if len(matches) >= 2:
            a = matches[0]
            b = matches[1]
            a_cypher_query=f"""
            MATCH (e) 
            WHERE e.name_en = '{a}'
            RETURN e.name
            """
            name_ac=link1.run(a_cypher_query)
            for ans in name_ac:
                if len(ans)==1:
                    a = str(ans[0]) 
            b_cypher_query=f"""
            MATCH (e) 
            WHERE e.name_en = '{b}'
            RETURN e.name
            """
            name_bc=link1.run(b_cypher_query)
            for ans in name_bc:
                if len(ans)==1:
                    b = str(ans[0]) 
            cypher_querya=f"""
            MATCH (n {{name: '{a}'}}) 
            RETURN labels(n)
            """
            answersa = link1.run(cypher_querya)
            linshi_answers = ""
            for ans in answersa:
                linshi_answers += str(ans[0])  
            matcha = re.search(r"\['([^]]+)'\]", linshi_answers)
            linshi_str = matcha.group(1) 
            cypher_queryb=f"""
            MATCH (n {{name: '{b}'}}) 
            RETURN labels(n)
            """
            answersb = link1.run(cypher_queryb)
            for ans in answersb:
                if isinstance(ans[0], list) and ans[0][0] == second_element:
                    name_ht=f"The name of the head entity is {a} and The name of the tail entity is {b}."
                    target1="Relationship detected"
            answersa = link1.run(cypher_querya)
            for ans in answersa:
                if isinstance(ans[0], list) and ans[0][0] == second_element:
                    name_ht=f"The name of the head entity is {a} and The name of the tail entity is {b}."
                    target1="Relationship detected"
        else:
            print("no data")

        cypher_query=f"""
        MATCH (n)
        WHERE n.name CONTAINS "{a}" AND n.name CONTAINS "{b}"
        RETURN labels(n),n.name limit 10
        """
        answers = link1.run(cypher_query)
        
        entity_answers = ""
        for ans in answers:
            entity_answers += str(ans[0]) 
            first_name += str(ans[1]) + ","
        match = re.search(r"\['([^]]+)'\]", entity_answers)
        if match:
            extracted_str = match.group(1)
            first_element = extracted_str 
        else:
            first_element=linshi_str
        
        name_pro=f"The specific name of the head entity is:{first_name}."
    attr_pro=f"The selected attribute is {third_element}."
    return first_pro, first_element, second_element, name_pro, attr_pro, target1,name_ht
