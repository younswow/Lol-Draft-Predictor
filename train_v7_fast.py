

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.special import expit

warnings.filterwarnings("ignore")
os.environ['OPENBLAS_NUM_THREADS'] = '1'

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


print("LOL DRAFT PREDICTOR V7 - FAST TRAINING")




print("\n[1] Loading data")

try:
    df_raw = pd.read_csv("lol_dataset_challenger_grandmaster_clean.csv", low_memory=False, encoding='utf-8')
except UnicodeDecodeError:
    df_raw = pd.read_csv("lol_dataset_challenger_grandmaster_clean.csv", low_memory=False, encoding='latin-1')

df_raw.columns = df_raw.columns.str.strip()


if 'match_ids' in df_raw.columns:
    df_raw = df_raw.rename(columns={'match_ids': 'match_id'})


if 'gameMode' in df_raw.columns:
    df_raw = df_raw[df_raw['gameMode'] == 'CLASSIC'].copy()

if 'teamId' in df_raw.columns:
    df_raw = df_raw[df_raw['teamId'].isin([100, 200])].copy()

df_raw['championName'] = df_raw['championName'].str.strip()

games = []
for match_id, group in df_raw.groupby('match_id'):
    t100 = group[group['teamId'] == 100]
    t200 = group[group['teamId'] == 200]
    
    if len(t100) == 5 and len(t200) == 5:
        win = int(float(t100['win'].iloc[0]) == 1.0)
        games.append({
            'team100': sorted(t100['championName'].tolist()),
            'team200': sorted(t200['championName'].tolist()),
            'win': win,
        })

games_df = pd.DataFrame(games)
print(f"   ✓ {len(games_df):,} games | Blue WR: {games_df['win'].mean():.1%}")


all_champs = sorted(set(c for game_list in games_df['team100'].tolist() + games_df['team200'].tolist() for c in game_list))
champ_to_idx = {c: i for i, c in enumerate(all_champs)}
print(f"   ✓ {len(all_champs)} unique champions")

# Train/test split
train_df, test_df = train_test_split(games_df, test_size=0.2, random_state=42)
print(f"   ✓ Train: {len(train_df):,} | Test: {len(test_df):,}")



print("\n[2] Engineering features")


emb_win = np.random.randn(len(all_champs), 16) * 0.1
emb_pop = np.random.randn(len(all_champs), 16) * 0.1

win_freq = np.zeros(len(all_champs))
pick_freq = np.zeros(len(all_champs))

for _, game in games_df.iterrows():
    for champ in game['team100']:
        champ_idx = champ_to_idx[champ]
        win_freq[champ_idx] += game['win']
        pick_freq[champ_idx] += 1
    for champ in game['team200']:
        champ_idx = champ_to_idx[champ]
        win_freq[champ_idx] += (1 - game['win'])
        pick_freq[champ_idx] += 1

win_rate = win_freq / (pick_freq + 1e-9)


for i in range(min(3, 16)):
    emb_win[:, i] = win_rate * 2.0
    emb_pop[:, i] = (pick_freq / pick_freq.max()) * 2.0

def extract_features(team100, team200):
    feat = []
    
    wr_blue = [win_rate[champ_to_idx[c]] for c in team100]
    wr_red = [win_rate[champ_to_idx[c]] for c in team200]
    feat.append(np.mean(wr_blue))
    feat.append(np.mean(wr_red))
    feat.append(np.mean(wr_blue) - np.mean(wr_red))
    
    syn_blue = [np.dot(emb_win[champ_to_idx[a]], emb_win[champ_to_idx[b]]) 
                for a, b in combinations(team100, 2)] or [0.0]
    syn_red = [np.dot(emb_win[champ_to_idx[a]], emb_win[champ_to_idx[b]]) 
               for a, b in combinations(team200, 2)] or [0.0]
    feat.append(np.mean(syn_blue))
    feat.append(np.mean(syn_red))
    feat.append(np.mean(syn_blue) - np.mean(syn_red))
    
    presence = [0.0] * len(all_champs)
    for c in team100:
        presence[champ_to_idx[c]] = 1.0
    for c in team200:
        presence[champ_to_idx[c]] = -1.0
    feat.extend(presence)
    
    return feat

print("extracting training features")
X_train = np.array([extract_features(team100, team200) for team100, team200 in zip(train_df['team100'], train_df['team200'])], dtype=np.float32)
y_train = train_df['win'].values

print("extracting test features")
X_test = np.array([extract_features(team100, team200) for team100, team200 in zip(test_df['team100'], test_df['team200'])], dtype=np.float32)
y_test = test_df['win'].values

print(f"Features: {X_train.shape} (train) | {X_test.shape} (test)")

# train models

print("\n[3] Training classifiers")

models = {
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=42))]),
    'Ridge': Pipeline([('scaler', StandardScaler()), ('ridge', LogisticRegression(penalty='l2', max_iter=1000, random_state=42))]),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
}

best_model = None
best_auc = 0
results = {}

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = expit(model.decision_function(X_test))
        
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))
        
        results[name] = {'auc': auc, 'acc': acc, 'model': model}
        print(f"   ✓ {name:<25} AUC={auc:.4f} | Acc={acc:.3f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = (name, model, auc)
    
    except Exception as e:
        print(f"   ✗ {name}: {str(e)[:40]}")

print(f"\n Best: {best_model[0]} (AUC={best_model[2]:.4f})")

#export models

print("\n[4] Exporting models")


with open(MODEL_DIR / 'classifier.pkl', 'wb') as f:
    pickle.dump(best_model[1], f)


export = {
    'emb_win': emb_win,
    'emb_pop': emb_pop,
    'all_champions': all_champs,
    'champ_to_idx': champ_to_idx,
    'win_rate': win_rate,
}

with open(MODEL_DIR / 'features.pkl', 'wb') as f:
    pickle.dump(export, f)


metadata = {
    'n_features': X_train.shape[1],
    'n_champions': len(all_champs),
    'model_type': best_model[0],
    'best_auc': float(best_auc),
    'created': str(pd.Timestamp.now()),
    'version': 'v7',
}

with open(MODEL_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  Classifier saved")
print(f"  Features saved")
print(f"  Metadata saved")

# validation

print("\n[5] Validating")

with open(MODEL_DIR / 'classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

with open(MODEL_DIR / 'features.pkl', 'rb') as f:
    feat = pickle.load(f)

test_pred = clf.predict_proba(X_test[:3])
print(f"Models load and predict successfully")
print(f"Sample predictions: {test_pred[0]}")
print("TRAINING FINISHED")
