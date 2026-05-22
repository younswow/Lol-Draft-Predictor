#!/usr/bin/env python3
# debug api
import sys
sys.path.insert(0, '.')

from ml_recommender import MLDraftRecommender
from champion_roles import get_missing_roles, get_champion_role

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

# simu api request
blue_picks = ["Malphite", "Vi", "Orianna", "Jinx"]
red_picks = ["Jayce", "Graves", "Zed", "Caitlyn"]
used = set(blue_picks + red_picks + ["Yasuo", "Garen"])  
available = [c for c in CHAMPIONS if c not in used]

print("Available champions:", len(available))
print("Missing roles:", get_missing_roles(blue_picks))
print()

recommender = MLDraftRecommender()
try:
    recs = recommender.recommend(blue_picks, red_picks, available, top_k=3)
    print("Recommendations returned:")
    for rec in recs:
        role = get_champion_role(rec.champion)
        print(f"  {rec.champion} ({role}) - {rec.win_probability:.1%}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
