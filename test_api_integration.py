#api integration test

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from ml_recommender import MLDraftRecommender, recommendations_to_api_format



print("\n[1] initializing")
recommender = MLDraftRecommender()


blue_team = ["Malphite", "Vi"]
red_team = ["Jayce"]
available_champs = [c for c in recommender.all_champions if c not in blue_team + red_team]

recs = recommender.recommend(
    blue_team=blue_team,
    red_team=red_team,
    available=available_champs,
    top_k=3
)

api_result = recommendations_to_api_format(recs)
print(f"\n  Blue team: {blue_team}")
print(f"  Red team: {red_team}")
print(f"  Top 3 recommendations:")
for rec in api_result:
    print(f"    - {rec['name']}: {rec['pct']}%")


draft_state = {
    "blue": {
        "picks": ["Malphite", "Vi", "Orianna"],
        "bans": ["Yasuo", "Azir"]
    },
    "red": {
        "picks": ["Jayce", "Graves"],
        "bans": ["Garen", "Sion"]
    }
}

used_champs = (
    draft_state["blue"]["picks"] + draft_state["blue"]["bans"] +
    draft_state["red"]["picks"] + draft_state["red"]["bans"]
)
available = [c for c in recommender.all_champions if c not in used_champs]

recs = recommender.recommend(
    blue_team=draft_state["blue"]["picks"],
    red_team=draft_state["red"]["picks"],
    available=available,
    top_k=3
)

api_format = recommendations_to_api_format(recs)
api_response = {
    "recommendations": api_format,
    "step": 9,
    "source": "ml_model"
}

print(f"\n  Request draft state:")
print(f"    Blue: {draft_state['blue']}")
print(f"    Red: {draft_state['red']}")
print(f"\n  API:")
print(f"  {json.dumps(api_response, indent=2)}")



blue_team = ["Malphite", "Vi", "Orianna", "Jinx", "Thresh"]
red_team = ["Jayce", "Graves", "Zed", "Caitlyn"]

available = [c for c in recommender.all_champions if c not in blue_team + red_team]
recs = recommender.recommend(
    blue_team=blue_team,
    red_team=red_team,
    available=available,
    top_k=3
)

print(f"\n  Blue: {blue_team}")
print(f"  R5 pick:")
for rec in recs:
    print(f"    - {rec.champion}: {rec.win_probability:.1%}")

