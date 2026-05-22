#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from ml_recommender import MLDraftRecommender
from champion_roles import get_champion_role

recommender = MLDraftRecommender()

#missing sup
blue_team = ["Malphite", "Vi", "Orianna", "Jinx"]  
red_team = ["Jayce", "Graves", "Zed", "Caitlyn"]

all_champs = [
    "Alistar", "Braum", "Leona", "Nautilus", "Thresh",  
    "Renekton", "Garen", "Darius",  
    "Amumu", "Khazix", "Sejuani", 
    "Ahri", "Zoe", "Viktor",  
    "Kaisa", "Samira", "Varus"  
]

available = [c for c in all_champs if c not in blue_team + red_team]

print("=" * 70)
print("TEST: Role Filtering (Should ONLY recommend SUP)")
print("=" * 70)
print(f"\nYour team: {blue_team}")
print(f"Roles: {[f'{c}({get_champion_role(c)})' for c in blue_team]}")
print(f"\nEnemy: {red_team}")
print(f"Missing role: SUP (need support)")
print(f"\nAvailable champions: {available}")
print("\nRecommendations (SHOULD BE ONLY SUPPORTS):\n")

recs = recommender.recommend(blue_team, red_team, available, top_k=5)

all_sup = True
for rec in recs:
    role = get_champion_role(rec.champion)
    print(f"  {rec.rank}. {rec.champion:15} ({role:3}) {rec.win_probability:5.1%}")
    if role != "SUP":
        all_sup = False

if all_sup:
    print("good : all reco are sup")
else:
    print("bad")
