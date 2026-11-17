# pip install peft
# for full instructions, see here:
# - on fine-tuning: https://huggingface.co/docs/transformers/training
# - on peft: https://huggingface.co/docs/peft/main/en/guides/peft_model_config 

# loading model and tokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# loading dataset
# gets just the first 2000 lines from the dataset
dataset = load_dataset("karthiksagarn/astro_horoscope", split="train[:2000]")

# data collation and tokenization
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# configuring PEFT

from peft import LoraConfig, TaskType
from peft import get_peft_model

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
lora_model = get_peft_model(model, lora_config)

# see our trainable params
lora_model.print_trainable_parameters()

# configure training params
training_args = TrainingArguments(
    output_dir="qwen3-finetuned",
    num_train_epochs=1, #start with one, then move up
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"every_n_layers": 4},
    bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
)
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# trains the model
trainer.train()

# save model
trainer.save_model('qwen-astro')

# load model back up
model = AutoModelForCausalLM.from_pretrained('./qwen-astro')
tokenizer = AutoTokenizer.from_pretrained('./qwen-astro')

# run inference 
from transformers import pipeline

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
pipe('Today will be')
