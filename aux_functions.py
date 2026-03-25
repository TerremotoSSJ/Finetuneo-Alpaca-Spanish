import torch



def format_text(text, boolean_train=False):
    system = "Eres un asistente útil que responde en español."
    instruction=text['instruction'] if 'instruction' in text else ""
    input_message=text['input'] if 'input' in text else ""

    input_text=f"{instruction}\n{input_message}" if instruction and input_message else instruction or input_message
    #output (if boolean_train is True)
    output=text['output'] if boolean_train and 'output' in text else ""
    
    formatted_text = f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
"""  
    if boolean_train:
        formatted_text+=f"""{output}<|im_end|>"""

    return formatted_text

def tokenize_function(examples, tokenizer):
# Input: dictionary with keys 'instruction', 'input', and 'output'

    formatted_texts = []
    for i in range(len(examples['instruction'])):
        text = {
            'instruction': examples['instruction'][i],
            'input': examples['input'][i],
            'output': examples['output'][i]
        }
        formatted_texts.append(format_text(text, boolean_train=True))
    
    tokenized = tokenizer(formatted_texts, truncation=True, max_length=128,padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def generate_answer(model, tokenizer, pregunta, contexto='', max_tokens=512):
    prompt=format_text({'instruction': pregunta, 'input': contexto}, boolean_train=False)
    

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate answers
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1
        )
    
    outputs= tokenizer.decode(outputs[0], skip_special_tokens=True)
    outputs = outputs.split("assistant\n")[-1].strip()
    return outputs