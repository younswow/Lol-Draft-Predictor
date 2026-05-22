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

app = Flask(__name__, static_folder=".")


 
#  front
@app.route("/")
def index():
    return send_from_directory(".", "draft_simulator.html")


# use ml prediction
@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    step = data.get("step", 0)
    draft = data.get("draft", {})

    blue = draft.get("blue", {})
    red  = draft.get("red",  {})
    
    
    used = set(
        filter(None, blue.get("bans", []) + blue.get("picks", []) +
                      red.get("bans",  []) + red.get("picks",  []))
    )

    
    CHAMPIONS = [
      "Aatrox","Ahri","Akali","Akshan","Ambessa","Amumu","Annie","Aphelios","Ashe","AurelionSol",
      "Azir","Bard","Blitzcrank","Brand","Braum","Briar","Caitlyn","Camille","Cassiopeia","Chogath",
      "Corki","Darius","Diana","DrMundo","Draven","Ekko","Elise","Evelynn","Ezreal","Fiora",
      "Fizz","Galio","Gangplank","Garen","Gnar","Gragas","Graves","Gwen","Hecarim","Heimerdinger",
      "Illaoi","Irelia","Ivern","JarvanIV","Jayce","Jhin","Jinx","Kaisa","Kalista","Kassadin",
      "Katarina","Kayle","Kayn","Kennen","Khazix","Kindred","KogMaw","KSante","Leblanc","LeeSin",
      "Leona","Lissandra","Lulu","Lux","Malphite","Maokai","Milio","MissFortune","Mordekaiser","Morgana",
      "Nami","Nautilus","Neeko","Nidalee","Nilah","Nocturne","Nunu","Olaf","Ornn","Pantheon",
      "Poppy","Pyke","Qiyana","Quinn","Rammus","RekSai","Renata","Renekton","Rengar","Riven",
      "Rumble","Ryze","Samira","Sejuani","Senna","Seraphine","Sett","Shaco","Shen","Shyvana",
      "Singed","Sion","Sivir","Skarner","Smolder","Sona","Soraka","Swain","Sylas","Syndra",
      "TahmKench","Taliyah","Talon","Taric","Teemo","Thresh","TwistedFate","Twitch","Udyr","Urgot",
      "Varus","Vayne","Veigar","Velkoz","Vex","Viktor","Vladimir","Volibear","Warwick","Wukong",
      "Xayah","Xerath","XinZhao","Yasuo","Yone","Yorick","Yuumi","Zac","Zed","Zeri",
      "Ziggs","Zilean","Zoe","Zyra"
    ]
    
    # filter champs already picked or banned
    available = [c for c in CHAMPIONS if c not in used]
    
    
    if ML_AVAILABLE and ML_RECOMMENDER:
        try:
            blue_picks = blue.get("picks", [])
            red_picks = red.get("picks", [])
            
            
            ml_recs = ML_RECOMMENDER.recommend(
                blue_team=blue_picks,
                red_team=red_picks,
                available=available,
                top_k=3
            )
            
            
            recommendations = recommendations_to_api_format(ml_recs)
            
            return jsonify({
                "recommendations": recommendations,
                "step": step,
                "source": "ml_model"
            })
        
        except Exception as e:
            import traceback
            print(f"ML recommendation failed: {e}")
            traceback.print_exc()
    
    # heuristic recommendations
    recommendations = generate_heuristic_recommendations(
        blue.get("picks", []),
        red.get("picks", []),
        step,
        available
    )
    
    return jsonify({
        "recommendations": recommendations,
        "step": step,
        "source": "heuristic"
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
