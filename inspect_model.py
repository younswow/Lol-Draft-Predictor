import pickle


with open('models/features.pkl', 'rb') as f:
    features = pickle.load(f)

print("Keys in features:")
for key in features.keys():
    val = features[key]
    if hasattr(val, 'shape'):
        print(f"  {key}: {type(val).__name__} shape {val.shape}")
    elif isinstance(val, dict):
        print(f"  {key}: dict with {len(val)} items")
    else:
        print(f"  {key}: {type(val).__name__}")


if 'lfm_item2id' in features:
    champ_map = features['lfm_item2id']
    print(f"\nChampion map ({len(champ_map)} items):")
    champs = sorted(list(champ_map.keys()))
    for c in champs[:20]:
        print(f"  {c}")
    if len(champs) > 20:
        print(f"  ... and {len(champs) - 20} more")
