#train models as we did for trainv7 pretty much it

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = Path('lol_dataset_challenger_grandmaster_clean.csv')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

print(f"Loading dataset: {DATA_PATH}")
print(f"Size: {DATA_PATH.stat().st_size / 1e9:.2f} GB")

try:
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
except:
    df = pd.read_csv(DATA_PATH, encoding='iso-8859-1')



all_champs = sorted(df['championName'].dropna().unique())


champ_idx = {champ: i for i, champ in enumerate(all_champs)}



try:
    if 'match_ids' in df.columns:
        df['match_key'] = df['match_ids']
    elif 'gameId' in df.columns:
        df['match_key'] = df['gameId']
    else:
        df['match_key'] = df.groupby('gameCreation').cumcount()
except:
    df['match_key'] = df.index // 10

duo_matrix = np.ones((len(all_champs), len(all_champs))) * 0.5
duo_counts = np.zeros((len(all_champs), len(all_champs)))
counter_matrix = np.ones((len(all_champs), len(all_champs))) * 0.5
counter_counts = np.zeros((len(all_champs), len(all_champs)))
champ_popularity = np.zeros(len(all_champs))

matches_processed = 0
for match_id, group in df.groupby('match_key'):
    if matches_processed % 1000 == 0:
        print(f"   Processed {matches_processed:,} matches")
    
    matches_processed += 1
    
    if len(group) != 10:
        continue
    
    try:
        blue_team = group[group['teamId'] == 100]['championName'].dropna().tolist()
        red_team = group[group['teamId'] == 200]['championName'].dropna().tolist()
        
        if len(blue_team) < 5 or len(red_team) < 5:
            continue
        
        result = group[group['teamId'] == 100]['win'].iloc[0] if len(group[group['teamId'] == 100]) > 0 else None
        if result is None or pd.isna(result):
            continue
        
        result = 1 if result else 0
        
        for b_champ in blue_team:
            if b_champ in champ_idx:
                b_idx = champ_idx[b_champ]
                champ_popularity[b_idx] += 1
                
                for other_b in blue_team:
                    if other_b != b_champ and other_b in champ_idx:
                        other_idx = champ_idx[other_b]
                        duo_matrix[b_idx, other_idx] += result
                        duo_counts[b_idx, other_idx] += 1
                
                for r_champ in red_team:
                    if r_champ in champ_idx:
                        r_idx = champ_idx[r_champ]
                        counter_matrix[b_idx, r_idx] += result
                        counter_counts[b_idx, r_idx] += 1
        
        for r_champ in red_team:
            if r_champ in champ_idx:
                r_idx = champ_idx[r_champ]
                champ_popularity[r_idx] += 1
                
                for other_r in red_team:
                    if other_r != r_champ and other_r in champ_idx:
                        other_idx = champ_idx[other_r]
                        duo_matrix[r_idx, other_idx] += (1 - result)
                        duo_counts[r_idx, other_idx] += 1
    
    except:
        continue


for i in range(len(all_champs)):
    for j in range(len(all_champs)):
        if duo_counts[i, j] > 0:
            duo_matrix[i, j] /= duo_counts[i, j]
        if counter_counts[i, j] > 0:
            counter_matrix[i, j] /= counter_counts[i, j]



from sklearn.decomposition import TruncatedSVD


svd_win = TruncatedSVD(n_components=32, random_state=42)
emb_win = svd_win.fit_transform(duo_matrix)


pop_matrix = np.diag(champ_popularity)
if pop_matrix.sum() > 0:
    svd_pop = TruncatedSVD(n_components=32, random_state=42)
    emb_pop = svd_pop.fit_transform(pop_matrix)
else:
    emb_pop = np.random.randn(len(all_champs), 32) * 0.1


macro_features = []

for champ in all_champs:
    champ_data = df[df['championName'] == champ]
    
    if len(champ_data) == 0:
        macro = np.random.rand(8) * 0.5 + 0.25
    else:
        wr = champ_data['win'].mean() if 'win' in champ_data.columns else 0.5
        popularity = len(champ_data) / len(df)
        
        dmg_dealt = champ_data['totalDamageDealtToChampions'].mean() if 'totalDamageDealtToChampions' in champ_data.columns else 0
        dmg_taken = champ_data['totalDamageTaken'].mean() if 'totalDamageTaken' in champ_data.columns else 0
        cc_dealt = champ_data['totalTimeCCDealt'].mean() if 'totalTimeCCDealt' in champ_data.columns else 0
        
        dmg_dealt_norm = np.clip(dmg_dealt / 10000, 0, 1) if dmg_dealt > 0 else 0.5
        dmg_taken_norm = np.clip(dmg_taken / 10000, 0, 1) if dmg_taken > 0 else 0.5
        cc_norm = np.clip(cc_dealt / 100, 0, 1) if cc_dealt > 0 else 0.5
        
        macro = np.array([
            wr,
            popularity,
            dmg_taken_norm,
            dmg_dealt_norm,
            cc_norm,
            0.5,
            0.5,
            0.5,
        ])
    
    macro_features.append(macro)

champ_style_vec = np.array(macro_features)




X_train = []
y_train = []

for match_id, group in df.groupby('match_key'):
    if len(X_train) % 1000 == 0:
        print(f"   Training data: {len(X_train):,} matches")
    
    if len(group) != 10:
        continue
    
    try:
        blue_team = group[group['teamId'] == 100]['championName'].dropna().tolist()
        red_team = group[group['teamId'] == 200]['championName'].dropna().tolist()
        
        if len(blue_team) < 5 or len(red_team) < 5:
            continue
        
        result = group[group['teamId'] == 100]['win'].iloc[0]
        if pd.isna(result):
            continue
        
        result = 1 if result else 0
        
        blue_syn = []
        for i, b1 in enumerate(blue_team):
            if b1 in champ_idx:
                for b2 in blue_team[i+1:]:
                    if b2 in champ_idx:
                        blue_syn.append(duo_matrix[champ_idx[b1], champ_idx[b2]])
        
        counter = []
        for b in blue_team:
            if b in champ_idx:
                for r in red_team:
                    if r in champ_idx:
                        counter.append(counter_matrix[champ_idx[b], champ_idx[r]])
        
        features = [
            np.mean(blue_syn) if blue_syn else 0.5,
            np.mean(counter) if counter else 0.5,
            1.0,
        ]
        
        X_train.append(features)
        y_train.append(result)
    
    except:
        continue

X_train = np.array(X_train)
y_train = np.array(y_train)



clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

score = clf.score(X_train, y_train)


models = {
    'classifier': clf,
    'emb_win': emb_win,
    'emb_pop': emb_pop,
    'champ_style_vec': champ_style_vec,
    'lfm_item2id': champ_idx,
    'duo_matrix': duo_matrix,
    'counter_matrix': counter_matrix,
}


with open(MODELS_DIR / 'classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open(MODELS_DIR / 'features.pkl', 'wb') as f:
    pickle.dump({k: v for k, v in models.items() if k != 'classifier'}, f)

metadata = {
    'n_champions': len(all_champs),
    'n_features': X_train.shape[1],
    'champion_list': all_champs,
    'classifier_accuracy': float(score),
    'created': pd.Timestamp.now().isoformat(),
}

with open(MODELS_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)


