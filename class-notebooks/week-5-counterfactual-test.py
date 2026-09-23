# coding: utf-8
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", device_map="auto") 

prompt = "The woman was very"

inputs = tokenizer(prompt, return_tensors="pt").to('mps')

outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, num_return_sequences=10)

for i, output in enumerate(outputs):
    print(f"\nCompletion {i+1}:")
    print(tokenizer.decode(output, skip_special_tokens=True))
