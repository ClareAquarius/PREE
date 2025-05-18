import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm

class Calculator:
    def __init__(self, model_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).eval().to(device)
        self.epsilon = 1e-8  # Numerical stability

    def get_distribution_and_entropy(self, prefix):
        """Return distribution of last token and cumulative entropy of the entire prefix"""
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids[0]  
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # (seq_len, vocab_size)
        
        # Calculate distribution of last token
        last_logits = logits[-1, :]
        last_probs = F.softmax(last_logits, dim=-1).float().cpu().numpy()
        
        # Calculate cumulative entropy
        cumulative_entropy = 0.0
        for i in range(logits.shape[0]):
            logit = logits[i]
            probs = F.softmax(logit, dim=-1).float()
            entropy = - (probs * torch.log(probs + self.epsilon)).sum()
            cumulative_entropy += entropy.item()
        
        return last_probs, cumulative_entropy

def optimize_selection(prefixes, N, calculator, alpha=0.5, beta=0.5):
    """Optimized prefix selection algorithm"""
    n = len(prefixes)
    
    # Precompute probability distributions and entropies for all prefixes
    print("Computing probability distributions and entropies...")
    distributions = []
    entropies = []
    for prefix in tqdm(prefixes):
        dist, entropy = calculator.get_distribution_and_entropy(prefix)
        distributions.append(dist)
        entropies.append(entropy)
    
    # Greedy algorithm selection
    selected = []
    remaining = set(range(n))
    
    # Initial selection
    first = np.argmin(entropies)
    selected.append(first)
    remaining.remove(first)
    current_cost = beta * entropies[first]
    
    # Iterative selection
    for _ in tqdm(range(N-1), desc="Selecting prefixes"):
        best_cost = float('inf')
        best_idx = -1
        
        if not remaining:
            raise ValueError("Not enough prefixes to select")
        
        for candidate in remaining:
            kl_cost = 0
            valid_kl = True
            for s in selected:
                p = distributions[s]
                q = distributions[candidate]
                
                # Numerical stability check
                if np.any(q <= 0):
                    valid_kl = False
                    break
                
                # Higher precision KL divergence calculation
                p_safe = np.clip(p, calculator.epsilon, 1.0)
                q_safe = np.clip(q, calculator.epsilon, 1.0)
                kl = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)))
                if np.isnan(kl) or np.isinf(kl):
                    valid_kl = False
                    break
                kl_cost += alpha * kl
            
            if not valid_kl:
                continue
                
            entropy_cost = beta * entropies[candidate]
            total_cost = current_cost + kl_cost + entropy_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_idx = candidate
        
        if best_idx == -1:
            available = list(remaining)
            best_idx = available[np.argmin([entropies[i] for i in available])]
        
        selected.append(best_idx)
        remaining.remove(best_idx)
        current_cost = best_cost
    
    return selected


if __name__ == "__main__":
    model_path = "/work/models/meta-llama/Llama-3-8B"
    input_file = "data/prefixes.txt"
    output_file = "data/selected_prefixes.txt"
    device = "cuda:2"

    calculator = Calculator(model_path, device)
    with open(input_file, "r") as f:
        prefixes = [line.strip() for line in f]
    
    selected_indices = optimize_selection(
        prefixes=prefixes,
        N=10,
        calculator=calculator,
        alpha=0.3,
        beta=0.5
    )
    
    with open(output_file, "w") as f:
        for idx in selected_indices:
            f.write(f"{prefixes[idx]}\n")
    
    print(f"Selected {len(selected_indices)} prefixes saved to {output_file}")