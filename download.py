from functools import partial
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from aux_functions import tokenize_function
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# Allow TensorFloat-32 on Ampere/Ada GPUs for faster matrix ops.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model_name = "Qwen/Qwen2.5-3B-Instruct"
model_base_dir = "./Qwen2.5-3B-Instruct-4bit"
new_model_dir = "./Qwen2.5-3B-Instruct-4bit-lora"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no esta disponible. Este entrenamiento seria demasiado lento en CPU.")

gpu_name = torch.cuda.get_device_name(0)
gpu_capability = torch.cuda.get_device_capability(0)
supports_bf16 = gpu_capability[0] >= 8
print(f"GPU detectada: {gpu_name} | capability: {gpu_capability} | bf16: {supports_bf16}")

# Configuration for 4-bit quantization
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                                  bnb_4bit_compute_dtype=torch.bfloat16 if supports_bf16 else torch.float16,
                                  bnb_4bit_use_double_quant=True)

#Configuration for LoRa
lora_config=LoraConfig(r=4,lora_alpha=8,
                       target_modules=["q_proj","k_proj","v_proj","o_proj"],
                       lora_dropout=0.1,task_type="CAUSAL_LM",bias="none")

tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  
tokenizer.padding_side = "right" 


dataset=load_dataset("bertin-project/alpaca-spanish")
train_data = dataset['train']  

# Ver estructura
print(f"Tipo: {type(train_data)}")
print(f"Columnas: {train_data.column_names}")
print(f"Total ejemplos: {len(train_data)}")
print(f"Features: {train_data.features}")




model=AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=quant_config,
    device_map="auto",
    dtype=torch.bfloat16 if supports_bf16 else torch.float16,
    low_cpu_mem_usage=True,
)
model.save_pretrained(model_base_dir)
tokenizer.save_pretrained(model_base_dir)

# 4 bit training preparation
model=prepare_model_for_kbit_training(model)

# LoRa training preparation
model=get_peft_model(model, lora_config)
model.config.use_cache = False


training_kwargs = {
    "output_dir": new_model_dir,
    "per_device_train_batch_size": 2,
    "num_train_epochs": 1,
    "optim": "paged_adamw_8bit",
    "fp16": not supports_bf16,
    "bf16": supports_bf16,
    "learning_rate": 2e-4,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 4,
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 2,
    "dataloader_num_workers": 0,  # ← 4 → 0
    "dataloader_pin_memory": False,  # ← True → False
    "remove_unused_columns": False,
    "warmup_steps": 100,
    "save_strategy": "steps",
}



training_args = TrainingArguments(**training_kwargs)

tokenize_with_tokenizer = partial(tokenize_function, tokenizer=tokenizer)
tokenized_dataset = train_data.map(tokenize_with_tokenizer, batched=True, remove_columns=train_data.column_names)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8),
)

trainer.train()

model.save_pretrained(new_model_dir)
tokenizer.save_pretrained(new_model_dir)





