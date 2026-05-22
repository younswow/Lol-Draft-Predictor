

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ml_recommender import MLDraftRecommender
from champion_roles import (
    get_champion_role, get_role_count, get_missing_roles, 
    get_role_priority_weight, is_team_complete
)


recommender = MLDraftRecommender()


print("\nearly draft , picking ADC when we need it")
print("-" * 80)

blue_team = ["Malphite", "Vi", "Orianna"]
red_team = ["Jayce", "Graves"]

print(f"Blue team: {blue_team}")
print(f"  Roles picked: {', '.join(f'{c}({get_champion_role(c)})' for c in blue_team)}")
print(f"  Role count: {get_role_count(blue_team)}")
print(f"  Missing roles: {get_missing_roles(blue_team)}")

print(f"\nRed team: {red_team}")
print(f"  Roles: {', '.join(f'{c}({get_champion_role(c)})' for c in red_team)}")


all_champs = [
    "Malphite", "Vi", "Orianna", "Jayce", "Graves", "Jinx", "Thresh", "Caitlyn",
    "Lux", "Renekton", "Ekko", "Ahri", "Braum", "Leona", "Nautilus"
]
available = [c for c in all_champs if c not in blue_team + red_team]

print(f"\nAvailable champions: {available}")


recs = recommender.recommend(
    blue_team=blue_team,
    red_team=red_team,
    available=available,
    top_k=5
)

print(f"\ntop 5 recommendations:")
for rec in recs:
    role = get_champion_role(rec.champion)
    role_weight = get_role_priority_weight(rec.champion, blue_team)
    role_priority = "we need it!!" if role_weight > 1.3 else ("might need it" if role_weight > 1 else "fine")
    print(f"  {rec.rank}. {rec.champion:15} ({role:3}) {rec.win_probability:5.1%} {role_priority}")



print("\nmid draft, filter duplicate roles")
print("-" * 80)

blue_team = ["Malphite", "Vi", "Orianna", "Jinx"]
red_team = ["Jayce", "Graves", "Zed", "Caitlyn"]

print(f"Blue team: {blue_team}")
print(f"  Roles: {', '.join(f'{c}({get_champion_role(c)})' for c in blue_team)}")
print(f"  Missing roles: {get_missing_roles(blue_team)}")

print(f"\nRed team: {red_team}")
print(f"  Roles: {', '.join(f'{c}({get_champion_role(c)})' for c in red_team)}")

available = [c for c in all_champs if c not in blue_team + red_team]
available.extend(["Thresh", "Leona", "Nautilus", "Braum", "Alistar"])
available = list(set(available))

recs = recommender.recommend(
    blue_team=blue_team,
    red_team=red_team,
    available=available[:10],  
    top_k=5
)

for rec in recs:
    role = get_champion_role(rec.champion)
    role_weight = get_role_priority_weight(rec.champion, blue_team)
    priority = "wnt" if role_weight > 1.4 else "mnt" if role_weight > 1 else "fine"
    print(f"  {rec.rank}. {rec.champion:15} ({role:3}) {rec.win_probability:5.1%} {priority}")



print("\ncounter pick enemy team")
print("-" * 80)

blue_team = ["Malphite"]
red_team = ["Jayce", "LeeSin", "Ahri"]

print(f"Our team: {blue_team}")
print(f"Enemy team: {red_team}")
print(f"  Enemy roles: {', '.join(f'{c}({get_champion_role(c)})' for c in red_team)}")

available = ["Thresh", "Leona", "Braum", "Alistar", "Nautilus", "Lux", "Annie"]


for champ in available[:5]:
    counter_score = recommender._calculate_counter_advantage(champ, red_team)
    synergy_score = recommender._calculate_team_synergy(champ, blue_team)
    print(f"  {champ:15} - Counter: {counter_score:.1%}  |  Synergy: {synergy_score:.1%}")



print("\nteam comp")
print("-" * 80)

teams = [
    ["Malphite", "Vi", "Orianna", "Jinx", "Thresh"],  
    ["Malphite", "Vi", "Orianna", "Jinx"],  
    ["Malphite", "Vi", "Orianna"], 
]

for team in teams:
    roles = get_role_count(team)
    missing = get_missing_roles(team)
    complete = is_team_complete(team)
    
    print(f"\nTeam: {team}")
    print(f"  Role distribution: {roles}")
    print(f"  Missing: {missing}")
    print(f"  Complete : {'y' if complete else 'n'}")



