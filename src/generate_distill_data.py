import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config_fftchain import get_args
import os
import gc

def generate_distill_data(args):
    print(f"[Generate distill data] Loading model from {args.model_path} on {args.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": args.device}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    prompts = [
        "The capital of France is",
        "Machine learning is a branch of",
        "In the future, technology will enable",
        "The meaning of life can be understood through",
        "Climate change affects our planet by",
        "Quantum computing represents a breakthrough in",
        "Artificial intelligence has transformed industries such as",
        "The human brain processes information by",
        "Deep learning models are trained using",
        "Natural language processing enables computers to",
        "Renewable energy sources include solar and",
        "The theory of relativity revolutionized our understanding of",
        "Genetic engineering allows scientists to modify",
        "Blockchain technology provides secure and decentralized",
        "Space exploration has led to discoveries about",
        "Neuroplasticity refers to the brain's ability to",
        "Photosynthesis is the process by which plants",
        "Economic systems can be categorized into",
        "The Industrial Revolution began in the late",
        "Democracy is a form of government where",
        "Evolution is driven by the process of",
        "The stock market fluctuates based on",
        "Programming languages such as Python are used for",
        "Ancient civilizations like the Egyptians built",
        "The water cycle involves evaporation condensation and"
    ]
    
    prompts = prompts * ((args.num_samples + len(prompts) - 1) // len(prompts))
    prompts = prompts[:args.num_samples]
    
    distill_data = []
    
    print(f"Generating {len(prompts)} distillation samples from layers {args.layer_start}-{args.layer_end}...")
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i+args.batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, output_hidden_states=True)
            
            target_layers_hidden = []
            for layer_idx in range(args.layer_start, args.layer_end + 1):
                target_layers_hidden.append(outputs.hidden_states[layer_idx + 1].cpu())
            
            distill_data.append({
                'input_ids': inputs['input_ids'].cpu(),
                'attention_mask': inputs['attention_mask'].cpu(),
                'target_hidden_states': target_layers_hidden,
                'logits': outputs.logits.cpu()
            })
            
            if (i + args.batch_size) % 40 == 0:
                print(f"Generate Progress: {min(i + args.batch_size, len(prompts))}/{len(prompts)}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, 'distill_data.pt')
    torch.save(distill_data, save_path)
    print(f"Saved {len(distill_data)} batches to {save_path}")
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = get_args()
    generate_distill_data(args)