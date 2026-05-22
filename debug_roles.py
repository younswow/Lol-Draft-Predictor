#!/usr/bin/env python3
#debug champs roles
from champion_roles import get_missing_roles, get_champion_role

blue_team = ["Malphite", "Vi", "Orianna", "Jinx"]

print("Blue team:", blue_team)
print("Roles picked:", [get_champion_role(c) for c in blue_team])
print("Missing roles:", get_missing_roles(blue_team))
