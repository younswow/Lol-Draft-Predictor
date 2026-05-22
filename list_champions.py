import pickle
import pandas as pd


with open('models/features.pkl', 'rb') as f:
    features = pickle.load(f)


champ_map = features.get('lfm_item2id', {})

if champ_map:
    champs = sorted([k for k in champ_map.keys() if isinstance(k, str)])
    print(f"Champions in model ({len(champs)}):")
    for c in champs:
        print(f"  {c}")
else:
    print("No champion map found")
