

import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)

MODEL_CACHE = {
    'classifier': MODEL_DIR / 'classifier.pkl',
    'features': MODEL_DIR / 'features.pkl',
    'metadata': MODEL_DIR / 'metadata.json',
}


def extract_models_from_notebook():
    
    print("Extracting models from notebook")
    

    models = {
        'emb_win': np.random.randn(160, 32),  
        'emb_pop': np.random.randn(160, 32),  
        'lfm_item2id': {i: f'champ_{i}' for i in range(160)},
        'champ_style_vec': np.random.randn(160, 8), 
        'classifier': None,  
    }
    
    return models


def create_dummy_models():
   
   
    from sklearn.ensemble import RandomForestClassifier
    
    
    n_samples = 100
    n_features = 50  
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)  
    
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    clf.fit(X_train, y_train)
    
    models = {
        'classifier': clf,
        'emb_win': np.random.randn(160, 32) * 0.1,
        'emb_pop': np.random.randn(160, 32) * 0.1,
        'champ_style_vec': np.random.randn(160, 8) * 0.1,
        'lfm_item2id': {i: f'champ_{i}' for i in range(160)},
    }
    
  
    with open(MODEL_CACHE['classifier'], 'wb') as f:
        pickle.dump(models['classifier'], f)
    
    with open(MODEL_CACHE['features'], 'wb') as f:
        pickle.dump({k: v for k, v in models.items() if k != 'classifier'}, f)
    
    metadata = {
        'n_features': n_features,
        'n_championsonsons': 160,
        'model_type': 'RandomForest',
        'created': str(pd.Timestamp.now()),
    }
    
    with open(MODEL_CACHE['metadata'], 'w') as f:
        json.dump(metadata, f)
    
    print(f"Models saved to {MODEL_DIR}")
    
    return models


def load_cached_models():
    
    print("loading trainned models")
    
    if not all(f.exists() for f in MODEL_CACHE.values()):
        missing = [f.name for f in MODEL_CACHE.values() if not f.exists()]
        print(f"Missing: {missing}")
        print("Falling back to synthetic models")
        return None
    
    try:
        print(f"Loading classifier from {MODEL_CACHE['classifier'].name}...")
        with open(MODEL_CACHE['classifier'], 'rb') as f:
            classifier = pickle.load(f)
        
        print(f"Loading features from {MODEL_CACHE['features'].name}...")
        with open(MODEL_CACHE['features'], 'rb') as f:
            features = pickle.load(f)
        
        print(f"Loading metadata from {MODEL_CACHE['metadata'].name}...")
        with open(MODEL_CACHE['metadata'], 'r') as f:
            metadata = json.load(f)
        
        models = {
            'classifier': classifier,
            'emb_win': features.get('emb_win'),
            'emb_pop': features.get('emb_pop'),
            'lfm_item2id': features.get('champ_to_idx', {}),
            'champ_style_vec': features.get('champ_style_vec'),
            'all_champions': features.get('all_champions', []),
            'win_rate': features.get('win_rate'),
        }
        
        print(f"Loaded trained model (v{metadata.get('version', '?')}) - AUC: {metadata.get('best_auc', '?'):.4f}")
        return models
    
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_models(force_create=False):
   
    if not force_create:
        models = load_cached_models()
        if models:
            return models
    
    models = create_dummy_models()
    
    return models


def validate_models(models):
    
    required = ['classifier', 'emb_win', 'emb_pop', 'champ_style_vec', 'lfm_item2id']
    
    for key in required:
        if key not in models:
            raise ValueError(f"Missing model component: {key}")
    
    # Quick sanity checks
    assert models['emb_win'].shape[0] > 0, "Embeddings empty"
    assert models['classifier'] is not None, "Classifier not loaded"
    
    print(f"All {len(required)} model components validated")
    return True


if __name__ == '__main__':
    models = load_models()
    validate_models(models)
    print(f"\n Model Summary:")
    print(f"Classifier type: {type(models['classifier']).__name__}")
    print(f"Embeddings shape: {models['emb_win'].shape}")
    print(f"Champions: {len(models['lfm_item2id'])}")
