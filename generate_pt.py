import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import numpy as np



# Define a helper function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = vec1.view(vec1.shape[0], -1)  # Flatten the tensors
    vec2 = vec2.view(vec2.shape[0], -1)
    sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=1)
    return sim.mean().item()

if __name__ == "__main__":
    device = torch.device(f'cuda:{5}')
    # Load the model and tokenizer
    model_path = "/home/sjx/pretrained_models/llama2-7b-chat-4k"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.ratio = 0.35
    # print(config)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval().to(device)
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).eval().to(device)

    prompt = """
    The "trusty system" (sometimes incorrectly called "trustee system") was a penitentiary system of discipline and security enforced in parts of the United States until the 1980s, in which designated inmates were given various privileges, abilities, and responsibilities not available to all inmates.It was made compulsory under Mississippi state law but was used in other states as well, such as Arkansas, Alabama, Louisiana, New York and Texas. The method of controlling and working inmates at Mississippi State Penitentiary at Parchman was designed in 1901 to replace convict leasing. The case Gates v. Collier ended the flagrant abuse of inmates under the trusty system and other prison abuses that had continued essentially unchanged since the building of the Mississippi State Penitentiary. Other states using the trusty system were also forced to give it up under the ruling.
    """
    
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    output = model.generate(
                **input,
                max_new_tokens=64,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
                return_dict_in_generate=True,
            )
    past_key_values = output.past_key_values
    torch.save(past_key_values, "llama2-7b-0.35.pt")
    

