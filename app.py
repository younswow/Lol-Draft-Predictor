"""

api endpoint for ML recommendations:
    POST /api/recommend
    Body: { "step": int, "draft": { blue: {...}, red: {...} } }
    Returns: { "recommendations": [{ "name": str, "pct": int }] }
"""

from flask import Flask, jsonify, request, send_from_directory
import os
import sys
import warnings

# ML recommender system
try:
    from ml_recommender import MLDraftRecommender, recommendations_to_api_format
    ML_RECOMMENDER = MLDraftRecommender()
    ML_AVAILABLE = True
    print("ML Recommender loaded")
except ImportError as e:
    print(f"ML recommender not loaded: {e}")
    ML_RECOMMENDER = None
    ML_AVAILABLE = False

import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# no static_folder: serving "." would expose the whole repo (code, model files)
app = Flask(__name__, static_folder=None)


#  front
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "draft_simulator.html")


# draft order (must match DRAFT_ORDER in draft_simulator.html)
DRAFT_ORDER = [
    ("blue", "ban"), ("red", "ban"), ("blue", "ban"), ("red", "ban"), ("blue", "ban"), ("red", "ban"),
    ("blue", "pick"), ("red", "pick"), ("red", "pick"), ("blue", "pick"), ("blue", "pick"),
    ("red", "ban"), ("blue", "ban"), ("red", "ban"), ("blue", "ban"),
    ("red", "pick"), ("blue", "pick"), ("blue", "pick"), ("red", "pick"), ("red", "pick"),
]

# single source of truth: the champions the model was trained on
if ML_AVAILABLE and ML_RECOMMENDER and ML_RECOMMENDER.all_champions:
    CHAMPIONS = list(ML_RECOMMENDER.all_champions)
else:
    from champion_roles import CHAMPION_ROLES
    CHAMPIONS = sorted(CHAMPION_ROLES)


@app.route("/api/champions")
def champions():
    from champion_roles import get_champion_roles
    return jsonify([{"name": c, "roles": list(get_champion_roles(c))} for c in CHAMPIONS])


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json() or {}
    step = int(data.get("step", 0))
    draft = data.get("draft", {})

    blue = draft.get("blue", {})
    red  = draft.get("red",  {})
    blue_picks = [c for c in blue.get("picks", []) if c]
    red_picks  = [c for c in red.get("picks", []) if c]

    used = set(
        filter(None, blue.get("bans", []) + blue.get("picks", []) +
                      red.get("bans",  []) + red.get("picks",  []))
    )
    available = [c for c in CHAMPIONS if c not in used]

    if step >= len(DRAFT_ORDER):
        return jsonify({"recommendations": [], "step": step, "source": "done"})

    team, action = DRAFT_ORDER[step]
    enemy = "red" if team == "blue" else "blue"
    # pick: best picks for the team on turn
    # ban: deny the best picks the ENEMY could make next
    target_side = team if action == "pick" else enemy

    if ML_AVAILABLE and ML_RECOMMENDER:
        try:
            ml_recs = ML_RECOMMENDER.recommend(
                blue_team=blue_picks,
                red_team=red_picks,
                available=available,
                top_k=3,
                side=target_side,
            )
            return jsonify({
                "recommendations": recommendations_to_api_format(ml_recs),
                "step": step,
                "team": team,
                "action": action,
                "source": "ml_model",
            })
        except Exception as e:
            import traceback
            print(f"ML recommendation failed: {e}")
            traceback.print_exc()

    from champion_roles import is_valid_addition
    ally = blue_picks if target_side == "blue" else red_picks
    available = [c for c in available if is_valid_addition(ally, c)]
    recommendations = generate_heuristic_recommendations(blue_picks, red_picks, step, available)
    return jsonify({
        "recommendations": recommendations,
        "step": step,
        "team": team,
        "action": action,
        "source": "heuristic",
    })


def generate_heuristic_recommendations(blue_picks, red_picks, step, available):
    
    if not available:
        return []
    
    # draft phase
    is_pick_phase = step >= 6 and step % 2 == (1 if step < 15 else 0)
    
    sample = random.sample(available, min(3, len(available)))
    
    
    if step < 8:  # early picks matter more
        w1 = 50 + random.randint(-10, 10)
        w2 = 30 + random.randint(-10, 10)
        w3 = max(5, 100 - w1 - w2)
    else:
        w1 = 40 + random.randint(-10, 10)
        w2 = 35 + random.randint(-10, 10)
        w3 = max(5, 100 - w1 - w2)
    
    return [
        {"name": sample[0], "pct": w1},
        {"name": sample[1], "pct": w2},
        {"name": sample[2], "pct": w3},
    ] if len(sample) >= 3 else [{"name": c, "pct": 33} for c in sample]


if __name__ == "__main__":
    print("LoL Draft Simulator with Global Prediction System")
    print("Backend Status:")
    print(f"   ML Recommender: {'Active' if ML_AVAILABLE else 'not Active'}")
    app.run(debug=True, port=5000)
