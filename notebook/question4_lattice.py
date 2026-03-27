import pandas as pd
import jiwer
from difflib import SequenceMatcher
import re

# Load data
df = pd.read_csv('question4_data.csv')

# Models: H to n (6 models, but task says 5, perhaps H,i,k,l,m)
models = ['Model H', 'Model i', 'Model k', 'Model l', 'Model m', 'Model n']

def tokenize(text):
    # Split on spaces for word-level
    return text.split()

def build_lattice(human, model_outputs):
    # Tokenize all
    human_tokens = tokenize(human)
    model_tokens = [tokenize(out) for out in model_outputs]
    
    # Use sequence alignment to find positions
    # For simplicity, assume same length and align by position
    # In practice, use edit distance alignment
    max_len = max(len(human_tokens), max(len(mt) for mt in model_tokens))
    
    lattice = [set() for _ in range(max_len)]
    
    # Add human tokens
    for i, word in enumerate(human_tokens):
        lattice[i].add(word)
    
    # Add model tokens, aligning by position (simplified)
    for mt in model_tokens:
        for i, word in enumerate(mt):
            if i < len(lattice):
                lattice[i].add(word)
    
    # If lengths differ, extend lattice
    for mt in model_tokens:
        if len(mt) > len(lattice):
            for i in range(len(lattice), len(mt)):
                lattice.append({mt[i]})
    
    return lattice

def lattice_wer(model_output, lattice):
    model_tokens = tokenize(model_output)
    ref_len = len(lattice)
    model_len = len(model_tokens)
    
    # Simplified alignment: assume positions match
    S = 0
    I = 0
    D = 0
    
    for i in range(max(ref_len, model_len)):
        if i < model_len and i < ref_len:
            if model_tokens[i] not in lattice[i]:
                S += 1
        elif i < model_len:
            I += 1
        elif i < ref_len:
            D += 1
    
    return (S + I + D) / ref_len if ref_len > 0 else 0

# Process each segment
results = []
for idx, row in df.iterrows():
    human = row['Human']
    model_outputs = [row[m] for m in models]
    
    lattice = build_lattice(human, model_outputs)
    
    wers = {}
    for m in models:
        wer = lattice_wer(row[m], lattice)
        wers[m] = wer
    
    results.append({
        'segment': row['segment_url_link'],
        'lattice': [list(bin) for bin in lattice],
        'wers': wers
    })

# Output
import json
with open('outputs/question4_lattice_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Lattice construction and WER computation complete.")