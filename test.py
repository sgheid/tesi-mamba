import torch
from transformers import MambaForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf", dtype=torch.float16, device_map="cuda")

prompt = "President Trump is"
ids = tok(prompt, return_tensors="pt")["input_ids"].to("cuda")

out = model.generate(
    ids,
    max_new_tokens=80,          
    do_sample=True,             
    temperature=0.8,            
    top_p=0.9,                  
    repetition_penalty=1.2,     
)

testo = tok.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(testo[:testo.rfind(".") + 1])   