from scipy.spatial.distance import cosine
def get_sentence_embedding(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def calculate_similarity(sentence1, sentence2, model, tokenizer):
    embedding1 = get_sentence_embedding(sentence1, model, tokenizer)
    embedding2 = get_sentence_embedding(sentence2, model, tokenizer)
    similarity = 1 - cosine(embedding1.squeeze(0).detach().numpy(), embedding2.squeeze(0).detach().numpy())
    return similarity
def exception_(old_question,pipeline1,tokenizer):
    prompt = """
    You are a reasoning robot, and you need to output natural language to answer my questions.
    For example:
    What drugs are ALK mutations in giant cell lung cancer resistant to?
    Output: ALK mutations in giant cell lung cancer are resistant to clotozantinib and luminaspib.
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
                out=line
                break  

    return out