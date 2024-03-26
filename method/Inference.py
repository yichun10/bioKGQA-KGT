def infer(old_question, pipeline1, tokenizer,result_string_with_separator):
    prompt = """
    You are a reasoning robot, and you need to perform the following two steps step by step: 1. Output a corresponding natural language sentence for each relationship chain. 2. Answer my question using natural language from step 1. 3.Translate all answers into English.
    Note: The output format is: Output: One sentence in natural language.
    For example:
    (ALK-p.L1196M-巨细胞肺癌)-[:resistance_to {evidence_level: 'case report'}]->(克唑替尼) 克唑替尼
    (ALK-p.C1156Y-巨细胞肺癌)-[:resistance_to {evidence_level: 'clinical trial - phase2'}]->(克唑替尼) 克唑替尼
    (ALK-p.F1174V-巨细胞肺癌)-[:resistance_to {evidence_level: 'clinical study'}]->(克唑替尼) 克唑替尼
    (ALK-p.C1156Y-巨细胞肺癌)-[:resistance_to {evidence_level: 'case report'}]->(luminespib) luminespib
    (ALK-p.F1245C-巨细胞肺癌)-[:resistance_to {evidence_level: 'case report'}]->(克唑替尼) 克唑替尼
    (CMTR1-ALK-巨细胞肺癌)-[:resistance_to {evidence_level: 'case report'}]->(克唑替尼) 克唑替尼
    What drugs are resistant to ALK in giant cell lung cancer?
    Output: ALK in giant cell lung cancer are resistant to clotozantinib and luminaspib.

    (cabozantinib)-[:treatment {fda_approved: true, nmpa_approved: false, score: '10'}]->(肾细胞癌) cabozantinib
    (伏罗尼布)-[:treatment {fda_approved: false, nmpa_approved: true, score: '10'}]->(肾细胞癌) 伏罗尼布
    (仑伐替尼)-[:treatment {fda_approved: true, nmpa_approved: true, score: '10'}]->(肾细胞癌) 仑伐替尼
    (纳武利尤单抗)-[:treatment {fda_approved: true, nmpa_approved: true, score: '10'}]->(肾细胞癌) 纳武利尤单抗
    (帕博利珠单抗)-[:treatment {fda_approved: true, nmpa_approved: true, score: '10'}]->(肾细胞癌) 帕博利珠单抗
    (伊匹木单抗)-[:treatment {fda_approved: true, nmpa_approved: true, score: '10'}]->(肾细胞癌) 伊匹木单抗
    (阿昔替尼)-[:treatment {fda_approved: true, nmpa_approved: true, score: '10'}]->(肾细胞癌) 阿昔替尼
    (tivozanib)-[:treatment {fda_approved: true, nmpa_approved: false, score: '10'}]->(肾细胞癌) tivozanib
    (temsirolimus)-[:treatment {fda_approved: true, nmpa_approved: false, score: '10'}]->(肾细胞癌) temsirolimus
    (替加氟)-[:treatment {fda_approved: false, nmpa_approved: true, score: '10'}]->(肾细胞癌) 替加氟
    What are the drug treatment options for renal cell carcinoma?
    Output: Renal cell carcinoma can be treated with the following drugs: cabozantinib, voronib, lenvatinib, nivolumab, pembrolizumab, ipilimumab, acitinib, tivozanib, temsirolimus, and tigafur.
    """
    input_text1 = prompt + result_string_with_separator+old_question
    sequences = pipeline1(
        input_text1,
        do_sample=True,
        top_k=10,
        return_full_text=False,
        top_p = 0.7, 
        temperature = 0.1, 
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=3000, 
    )
    for seq in sequences:
        lines = seq['generated_text'].split('\n')
        for line in lines:
            if line.strip(): 
                out=line
                break  
    return out