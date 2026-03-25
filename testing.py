import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from aux_functions import format_text, generate_answer
from peft import PeftModel



# Loading base model
base_model = AutoModelForCausalLM.from_pretrained(
    "./Qwen2.5-3B-Instruct-4bit",  
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)


#Loading fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-3B-Instruct-4bit-lora")
tokenizer.pad_token = tokenizer.eos_token

fine_tuned_model = PeftModel.from_pretrained(
    base_model, 
    "./Qwen2.5-3B-Instruct-4bit-lora"
)






#Testing the models
print("\n" + "="*50)
print("PROBANDO MODELO")
print("="*50)


preguntas=[
    "Genera una nueva oración para comparar los niveles de corrupción en América del Norte y Oriente Medio.",
    "Redacta un tweet resumiendo la película 'The Matrix' en 140 caracteres.",
    "Explica las implicaciones éticas de construir un sistema de atención médica controlado por inteligencia artificial",
    "Genera algunas preguntas relevantes para hacer sobre el siguiente tema",
    "Crear un titular de noticias para una historia sobre una celebridad que acaba de publicar un libro"
]

contextos=[
    "",
    "",
    "",
    "Los avances en tecnología",
    "Tom Hanks"
]


# Prueba 1

for index in range(len(preguntas)):
    print(f"\n📝 Pregunta {index+1}:  " + preguntas[index]+"\nContexto: " + contextos[index])
    print(f"Fine tuned model:\n")
    print(f"🤖 Respuesta: {generate_answer(fine_tuned_model, tokenizer, pregunta=preguntas[index], contexto=contextos[index])}")
    print(f"Base model:\n")
    print(f"🤖 Respuesta: {generate_answer(base_model, tokenizer, pregunta=preguntas[index], contexto=contextos[index])}")

