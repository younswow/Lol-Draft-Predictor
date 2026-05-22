

import os
import sys
import pickle
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from scipy import sparse as sp
from scipy.special import expit

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet, Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ['OPENBLAS_NUM_THREADS'] = '1'



CSV_PATH = "lol_dataset_challenger_grandmaster_clean.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

MIN_GAMES_FOR_STATS = 20
MIN_CHAMP_GAMES = 200
TEST_SIZE = 0.20
RANDOM_STATE = 42
LFM_COMPONENTS = 32
LFM_EPOCHS = 30
LFM_ALPHA = 1.0

print("\n Loading data")

try:
    df_raw = pd.read_csv(CSV_PATH, low_memory=False, encoding='utf-8')
except UnicodeDecodeError:
    
    df_raw = pd.read_csv(CSV_PATH, low_memory=False, encoding='latin-1')

df_raw.columns = df_raw.columns.str.strip()

if 'match_ids' in df_raw.columns:
    df_raw = df_raw.rename(columns={'match_ids': 'match_id'})

if 'gameMode' in df_raw.columns:
    df_raw = df_raw[df_raw['gameMode'] == 'CLASSIC'].copy()

if 'teamId' in df_raw.columns:
    df_raw = df_raw[df_raw['teamId'].isin([100, 200])].copy()

df_raw['championName'] = df_raw['championName'].str.strip()

print(f"Rows: {len(df_raw):,} Unique games: {df_raw['match_id'].nunique():,}")

games = []
for match_id, group in df_raw.groupby('match_id'):
    t100 = group[group['teamId'] == 100]
    t200 = group[group['teamId'] == 200]
    
    if len(t100) != 5 or len(t200) != 5:
        continue
    
    win_val = t100['win'].iloc[0]
    if isinstance(win_val, bool):
        win = 1 if win_val else 0
    elif isinstance(win_val, str):
        win = 1 if win_val.lower() in ['true', 'win', '1'] else 0
    else:
        win = int(win_val)
    
    games.append({
        'match_id': match_id,
        'team100_champs': sorted(t100['championName'].tolist()),
        'team200_champs': sorted(t200['championName'].tolist()),
        'win': win,
    })

games_df = pd.DataFrame(games)
print(f"Games aggregated: {len(games_df):,} Blue WR: {games_df['win'].mean():.1%}")

# Train/test split
train_df, test_df = train_test_split(games_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Train: {len(train_df):,} Test: {len(test_df):,}")


print("\nBuilding LightFM embeddings")

all_champs = sorted(set(
    champ for game_list in games_df['team100_champs'].tolist() +
                            games_df['team200_champs'].tolist()
    for champ in game_list
))
n_champs = len(all_champs)
champ_to_idx = {c: i for i, c in enumerate(all_champs)}

print(f"{n_champs} unique champions")

#interaction matrix, win pop
row_idx_pop, col_idx_pop, data_pop = [], [], []
row_idx_win, col_idx_win, data_win = [], [], []

for idx, (_, game) in enumerate(games_df.iterrows()):
    #blue
    for champ in game['team100_champs']:
        champ_idx = champ_to_idx[champ]
        row_idx_pop.append(idx)
        col_idx_pop.append(champ_idx)
        data_pop.append(1.0)
        
        row_idx_win.append(idx)
        col_idx_win.append(champ_idx)
        data_win.append(1.0 + LFM_ALPHA * game['win'])
    
    #red
    for champ in game['team200_champs']:
        champ_idx = champ_to_idx[champ]
        row_idx_pop.append(idx + len(games_df))
        col_idx_pop.append(champ_idx)
        data_pop.append(1.0)
        
        row_idx_win.append(idx + len(games_df))
        col_idx_win.append(champ_idx)
        data_win.append(1.0 + LFM_ALPHA * (1 - game['win']))

interactions_pop = sp.coo_matrix(
    (data_pop, (row_idx_pop, col_idx_pop)), 
    shape=(2 * len(games_df), n_champs)
).tocsr()

interactions_win = sp.coo_matrix(
    (data_win, (row_idx_win, col_idx_win)),
    shape=(2 * len(games_df), n_champs)
).tocsr()


np.random.seed(RANDOM_STATE)
emb_pop = np.random.randn(n_champs, LFM_COMPONENTS) * 0.1
popularity = np.array(interactions_pop.sum(axis=0)).flatten()
popularity_norm = popularity / (popularity.sum() + 1e-9)
for i in range(min(5, LFM_COMPONENTS)):
    emb_pop[:, i] += popularity_norm * 2.0

# emb for winning
emb_win = np.random.randn(n_champs, LFM_COMPONENTS) * 0.1
win_freq = np.zeros(n_champs)
for game_idx, (_, game) in enumerate(games_df.iterrows()):
    for champ in game['team100_champs']:
        champ_idx = champ_to_idx[champ]
        win_freq[champ_idx] += game['win']
    for champ in game['team200_champs']:
        champ_idx = champ_to_idx[champ]
        win_freq[champ_idx] += (1 - game['win'])

win_rate = win_freq / (popularity + 1e-9)
for i in range(min(5, LFM_COMPONENTS)):
    emb_win[:, i] += win_rate * 2.0


print("\nBuilding macro style vectors")

style_cols = ['obj_control', 'siege', 'teamfight', 'frontline', 'dps', 'cc_density', 'vision_per_min', 'early_pressure']
champ_style_vec = pd.DataFrame(
    np.random.randn(n_champs, len(style_cols)) * 0.1,
    index=all_champs,
    columns=style_cols
)



print("\nLFM")

class DraftFeatureEngineering:
   
    
    def __init__(self, min_games, emb_win, emb_pop, lfm_item2id, champ_style_vec):
        self.min_games = min_games
        self.emb_win = emb_win
        self.emb_pop = emb_pop
        self.lfm_item2id = lfm_item2id
        self.champ_style_vec = champ_style_vec
        self.all_champions = list(lfm_item2id.keys())
        self.n_champs = len(self.all_champions)
        self.champion_to_idx = {c: i for i, c in enumerate(self.all_champions)}
        self.fitted = False
        self.synergy_wr = defaultdict(lambda: 0.0)
    
    def fit(self, df):
        for _, game in df.iterrows():
            team100 = game['team100_champs']
            team200 = game['team200_champs']
            outcome = game['win']
            
            for a, b in combinations(team100, 2):
                pair = tuple(sorted([a, b]))
                self.synergy_wr[pair] += outcome
            
            for a, b in combinations(team200, 2):
                pair = tuple(sorted([a, b]))
                self.synergy_wr[pair] += (1 - outcome)
        
        self.fitted = True
        return self
    
    def _extract(self, c100, c200):
        
        if not all(c in self.lfm_item2id for c in c100 + c200):
            return None
        
        feat = []
        
        # FM synergies
        fm100 = [np.dot(self.emb_win[self.lfm_item2id[a]], self.emb_win[self.lfm_item2id[b]])
                 for a, b in combinations(c100, 2)] or [0.0]
        fm200 = [np.dot(self.emb_win[self.lfm_item2id[a]], self.emb_win[self.lfm_item2id[b]])
                 for a, b in combinations(c200, 2)] or [0.0]
        
        feat += [np.mean(fm100), np.mean(fm200), np.mean(fm100) - np.mean(fm200)]
        
        # uplift (win - pop)
        up100 = [
            np.dot(self.emb_win[self.lfm_item2id[a]], self.emb_win[self.lfm_item2id[b]]) -
            np.dot(self.emb_pop[self.lfm_item2id[a]], self.emb_pop[self.lfm_item2id[b]])
            for a, b in combinations(c100, 2)
        ] or [0.0]
        up200 = [
            np.dot(self.emb_win[self.lfm_item2id[a]], self.emb_win[self.lfm_item2id[b]]) -
            np.dot(self.emb_pop[self.lfm_item2id[a]], self.emb_pop[self.lfm_item2id[b]])
            for a, b in combinations(c200, 2)
        ] or [0.0]
        
        feat += [np.mean(up100), np.mean(up200), np.mean(up100) - np.mean(up200)]
        
        for col in style_cols:
            cov100 = sum(self.champ_style_vec.loc[c, col] for c in c100)
            cov200 = sum(self.champ_style_vec.loc[c, col] for c in c200)
            feat += [cov100, cov200, cov100 - cov200]
        
        presence = [0.0] * self.n_champs
        for c in c100:
            presence[self.champion_to_idx[c]] = 1.0
        for c in c200:
            presence[self.champion_to_idx[c]] = -1.0
        feat += presence
        
        return feat
    
    def transform(self, df):
       
        rows = []
        valid_indices = []
        for idx, (_, r) in enumerate(df.iterrows()):
            feat = self._extract(r['team100_champs'], r['team200_champs'])
            if feat is not None:
                rows.append(feat)
                valid_indices.append(idx)
        return np.array(rows, dtype=np.float32), valid_indices

fe = DraftFeatureEngineering(MIN_GAMES_FOR_STATS, emb_win, emb_pop, champ_to_idx, champ_style_vec)
fe.fit(train_df)

X_train, idx_train = fe.transform(train_df)
y_train = train_df['win'].values[idx_train]

X_test, idx_test = fe.transform(test_df)
y_test = test_df['win'].values[idx_test]


print("\n[5/7] Training classifiers")

models_to_train = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    'Ridge': Pipeline([('scaler', StandardScaler()), ('ridge', RidgeClassifier(random_state=RANDOM_STATE))]),
    'LDA': LinearDiscriminantAnalysis(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
}

best_model = None
best_auc = 0
results = {}

for name, model in models_to_train.items():
    try:
        model.fit(X_train, y_train)
        
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = model.decision_function(X_test)
            y_pred = expit(y_pred)
        
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))
        
        results[name] = {'auc': auc, 'acc': acc, 'model': model}
        print(f"{name:<25} AUC={auc:.4f} | Acc={acc:.3f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = (name, model, auc)
    
    except Exception as e:
        print(f"{name}: {str(e)[:50]}")

print(f"\n Best: {best_model[0]} (AUC={best_model[2]:.4f})")



export_data = {
    'classifier': best_model[1],
    'emb_win': emb_win,
    'emb_pop': emb_pop,
    'champ_style_vec': champ_style_vec,
    'lfm_item2id': champ_to_idx,
    'all_champions': all_champs,
    'feature_engineering': fe,
    'training_results': results,
}


with open(MODEL_DIR / 'classifier.pkl', 'wb') as f:
    pickle.dump(best_model[1], f)
print(f"Classifier: {MODEL_DIR / 'classifier.pkl'}")


features_data = {k: v for k, v in export_data.items() if k != 'classifier'}
with open(MODEL_DIR / 'features.pkl', 'wb') as f:
    pickle.dump(features_data, f)
print(f"Features: {MODEL_DIR / 'features.pkl'}")


metadata = {
    'n_features': X_train.shape[1],
    'n_champions': n_champs,
    'model_type': best_model[0],
    'best_auc': float(best_auc),
    'train_size': len(train_df),
    'test_size': len(test_df),
    'created': str(pd.Timestamp.now()),
    'version': 'v7',
}

with open(MODEL_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)


with open(MODEL_DIR / 'classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)

with open(MODEL_DIR / 'features.pkl', 'rb') as f:
    loaded_feat = pickle.load(f)


test_pred = loaded_clf.predict_proba(X_test[:5])

