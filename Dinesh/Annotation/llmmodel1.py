from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "You are a helpful assistant.\nUser: What is overfitting in machine learning?\nAssistant:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
