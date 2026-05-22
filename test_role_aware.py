import requests
import json


draft_state = {
    'step': 10,
    'draft': {
        'blue': {'picks': ['Malphite', 'LeeSin', 'Ahri'], 'bans': []},
        'red': {'picks': ['Jayce', 'Graves'], 'bans': []}
    }
}

print("[*] Testing role-aware recommendations...")
print(f"Blue picks: {draft_state['draft']['blue']['picks']}")
print(f"Red picks: {draft_state['draft']['red']['picks']}")
print()

try:
    response = requests.post('http://localhost:5000/api/recommend', json=draft_state, timeout=5)
    data = response.json()
    
    print("[OK] API Response:")
    print(json.dumps(data, indent=2))
    
    recs = data.get('recommendations', [])
    print(f"\n[*] Got {len(recs)} recommendations:")
    for rec in recs:
        print(f"  - {rec.get('name')} ({rec.get('role')}) - {rec.get('pct')}% win")

except Exception as e:
    print(f"[ERROR] {e}")
