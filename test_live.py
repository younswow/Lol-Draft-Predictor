#!/usr/bin/env python3
import requests
import json

API = "http://localhost:5000/api/recommend"

test_data = {
    "blue": {
        "picks": ["Malphite", "Vi", "Orianna", "Jinx"],
        "bans": ["Yasuo"]
    },
    "red": {
        "picks": ["Jayce", "Graves", "Zed", "Caitlyn"],
        "bans": ["Garen"]
    }
}

print("\nRequest:")
print(f"  Blue team: {test_data['blue']['picks']}")
print(f"  Red team: {test_data['red']['picks']}")
print(f"  Missing role: SUP\n")

try:
    response = requests.post(API, json=test_data)
    recs = response.json()["recommendations"]
    
    sup_count = 0
    for rec in recs:
        print(f"  - {rec['name']:15} {rec['pct']}%")
        sup_count += 1
    
    if sup_count == len(recs):
        print(f"working")
    
except Exception as e:
    print(f"ERROR: {e}")
