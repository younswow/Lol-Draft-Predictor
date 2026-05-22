#!/usr/bin/env python3
#debug filtering logic
import sys
sys.path.insert(0, '.')

from ml_recommender import ROLES_AVAILABLE
from champion_roles import get_champion_role, get_missing_roles

print(f"ROLES_AVAILABLE: {ROLES_AVAILABLE}")
print(f"get_champion_role imported: {callable(get_champion_role)}")
print(f"get_missing_roles imported: {callable(get_missing_roles)}")

# test
blue_team = ["Malphite", "Vi", "Orianna", "Jinx"]
missing_roles = get_missing_roles(blue_team)

available = ["Thresh", "Nautilus", "Katarina", "Ekko", "Braum"]

print(f"\nBlue team: {blue_team}")
print(f"Missing roles: {missing_roles}")
print(f"Available: {available}")

if ROLES_AVAILABLE and missing_roles:
    candidates_by_missing_roles = [c for c in available if get_champion_role(c) in missing_roles]
    print(f"Filtered to missing roles: {candidates_by_missing_roles}")
else:
    print(f"Filtering not applied! ROLES_AVAILABLE={ROLES_AVAILABLE}, missing={missing_roles}")
