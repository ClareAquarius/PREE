import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_embedding(text):
    """Get text embedding vector"""
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    last_hidden = outputs.hidden_states[-1]
    attention_mask = inputs.attention_mask.unsqueeze(-1)
    return (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

def calculate_ppl(text):
    """Calculate text perplexity"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

if __name__ == "__main__":
    model_path = "/work/models/meta-llama/Llama-3-8B"
    prefix_file = "data/selected_prefixes.txt"
    input_file = "data/prepared_data.json"
    output_file1 = "data/train_data.json"
    output_file2 = "data/val_data.json"
    
    target_output = "Virendale"
    LAMBDA = 0.5 
    device = "cuda:1"

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).eval().to(device)

    # Read prefix list
    with open(prefix_file, "r") as f:
        prefixes = [line.strip() for line in f]

    # Load data file
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # Save original data (for training set)
    original_data = [item.copy() for item in data]

    # Find new prefixes for knowledge pairs (for validation set)
    for item in data:
        rewrite = item["requested_rewrite"]
        filled_prompt = rewrite["prompt"].format(rewrite["subject"])

        prompt = rewrite["prompt"]
        best_score = -float("inf")
        best_prefix = ""
        orig_embedding = get_embedding(filled_prompt)
        
        for prefix in prefixes:
            combined = f"{prefix} {filled_prompt}"  
            
            try:
                # Calculate cosine similarity
                comb_embedding = get_embedding(combined)
                cos_sim = F.cosine_similarity(orig_embedding, comb_embedding).item()
                
                # Calculate perplexity score
                ppl = calculate_ppl(combined)
                inv_ppl = 1.0 / ppl if ppl != 0 else 0.0
                
                # Composite score
                score = (1-LAMBDA)*cos_sim + LAMBDA*inv_ppl
                
                if score > best_score:
                    best_score = score
                    best_prefix = prefix
                    
            except Exception as e:
                print(f"Error processing prefix {prefix}: {str(e)}")
                continue
        
        # Save best prefix
        item["requested_rewrite"]["prompt"] = best_prefix + item["requested_rewrite"]["prompt"]
        item["requested_rewrite"]["target_new"]["str"] = target_output
    
    # Save validation set results (modified data)
    with open(output_file2, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Combine original correct knowledge pairs with newly constructed ones (for training set)
    combined_data = original_data + data
    
    # Save training set results (original + modified data)
    with open(output_file1, "w") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)