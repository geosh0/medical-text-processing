import numpy as np
import torch
import gc
from tqdm import tqdm
from torch.amp import autocast

def get_doc_vec(tokens, model):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)

def get_glove_vec(text, model, dim):
    tokens = text.split()
    valid_vectors = [model[word] for word in tokens if word in model]
    if not valid_vectors:
        return np.zeros(dim)
    return np.mean(valid_vectors, axis=0)

def get_biobert_embeddings_memory_safe(texts, model, tokenizer, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_embeddings =[]
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BioBERT Embeddings"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            with autocast(device_type=device.type):
                outputs = model(**inputs)
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().to(torch.float16).numpy()
        all_embeddings.append(cls_embeddings)
        
        if (i // batch_size) % 100 == 0:
            del outputs, inputs
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
                
    return np.vstack(all_embeddings)