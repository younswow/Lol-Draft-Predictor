"""
Champion roles and draft-coherence rules.

Each champion has a list of playable roles (first = main role). Flex picks
(Gragas, Pantheon, Seraphine...) list several roles.

A team is coherent if its picks can be assigned to DISTINCT roles
(TOP, JGL, MID, ADC, SUP), each champion on a role it can play.
A candidate is only recommendable if adding it keeps the team coherent.
So, with 4 picks covering TOP/JGL/MID/SUP, only champions that can play ADC
remain valid, while flex picks stay open as long as possible.
"""

from itertools import permutations
from typing import Dict, List, Optional, Tuple

ROLES = ["TOP", "JGL", "MID", "ADC", "SUP"]

# champion -> playable roles, main role first.
# Names follow Riot's match-v5 championName (same as the model's vocabulary).
CHAMPION_ROLES: Dict[str, Tuple[str, ...]] = {
    "Aatrox": ("TOP",), "Ahri": ("MID",), "Akali": ("MID", "TOP"), "Akshan": ("MID", "TOP"),
    "Alistar": ("SUP",), "Ambessa": ("TOP", "MID", "JGL"), "Amumu": ("JGL", "SUP"),
    "Anivia": ("MID",), "Annie": ("MID", "SUP"), "Aphelios": ("ADC",), "Ashe": ("ADC", "SUP"),
    "AurelionSol": ("MID",), "Aurora": ("MID", "TOP"), "Azir": ("MID",), "Bard": ("SUP",),
    "Belveth": ("JGL",), "Blitzcrank": ("SUP",), "Brand": ("SUP", "MID", "JGL"), "Braum": ("SUP",),
    "Briar": ("JGL",), "Caitlyn": ("ADC",), "Camille": ("TOP", "SUP"), "Cassiopeia": ("MID", "TOP"),
    "Chogath": ("TOP", "MID"), "Corki": ("MID", "ADC"), "Darius": ("TOP",), "Diana": ("JGL", "MID"),
    "DrMundo": ("TOP", "JGL"), "Draven": ("ADC",), "Ekko": ("JGL", "MID"), "Elise": ("JGL", "SUP"),
    "Evelynn": ("JGL",), "Ezreal": ("ADC",), "FiddleSticks": ("JGL", "SUP"), "Fiora": ("TOP",),
    "Fizz": ("MID",), "Galio": ("MID", "SUP"), "Gangplank": ("TOP", "MID"), "Garen": ("TOP",),
    "Gnar": ("TOP",), "Gragas": ("JGL", "TOP", "MID", "SUP"), "Graves": ("JGL",), "Gwen": ("TOP", "JGL"),
    "Hecarim": ("JGL",), "Heimerdinger": ("MID", "SUP", "TOP"), "Hwei": ("MID", "SUP"),
    "Illaoi": ("TOP",), "Irelia": ("TOP", "MID"), "Ivern": ("JGL",), "Janna": ("SUP",),
    "JarvanIV": ("JGL",), "Jax": ("TOP", "JGL"), "Jayce": ("TOP", "MID"), "Jhin": ("ADC",),
    "Jinx": ("ADC",), "KSante": ("TOP",), "Kaisa": ("ADC",), "Kalista": ("ADC",),
    "Karma": ("SUP", "MID", "TOP"), "Karthus": ("JGL",), "Kassadin": ("MID",), "Katarina": ("MID",),
    "Kayle": ("TOP", "MID"), "Kayn": ("JGL",), "Kennen": ("TOP", "MID"), "Khazix": ("JGL",),
    "Kindred": ("JGL",), "Kled": ("TOP",), "KogMaw": ("ADC",), "Leblanc": ("MID",),
    "LeeSin": ("JGL",), "Leona": ("SUP",), "Lillia": ("JGL", "TOP"), "Lissandra": ("MID", "TOP"),
    "Lucian": ("ADC", "MID"), "Lulu": ("SUP",), "Lux": ("SUP", "MID"), "Malphite": ("TOP", "SUP", "MID"),
    "Malzahar": ("MID",), "Maokai": ("SUP", "TOP", "JGL"), "MasterYi": ("JGL",), "Mel": ("MID", "SUP"),
    "Milio": ("SUP",), "MissFortune": ("ADC",), "MonkeyKing": ("JGL", "TOP"),
    "Mordekaiser": ("TOP", "JGL"), "Morgana": ("SUP", "MID"), "Naafiri": ("MID", "JGL"), "Nami": ("SUP",),
    "Nasus": ("TOP",), "Nautilus": ("SUP",), "Neeko": ("MID", "SUP"), "Nidalee": ("JGL",),
    "Nilah": ("ADC",), "Nocturne": ("JGL",), "Nunu": ("JGL",), "Olaf": ("TOP", "JGL"), "Orianna": ("MID",),
    "Ornn": ("TOP",), "Pantheon": ("TOP", "MID", "SUP", "JGL"), "Poppy": ("JGL", "TOP", "SUP"),
    "Pyke": ("SUP",), "Qiyana": ("MID", "JGL"), "Quinn": ("TOP",), "Rakan": ("SUP",), "Rammus": ("JGL",),
    "RekSai": ("JGL",), "Rell": ("SUP",), "Renata": ("SUP",), "Renekton": ("TOP",),
    "Rengar": ("JGL", "TOP"), "Riven": ("TOP",), "Rumble": ("TOP", "MID", "JGL"), "Ryze": ("MID", "TOP"),
    "Samira": ("ADC",), "Sejuani": ("JGL",), "Senna": ("SUP", "ADC"), "Seraphine": ("SUP", "MID", "ADC"),
    "Sett": ("TOP", "SUP"), "Shaco": ("JGL", "SUP"), "Shen": ("TOP", "SUP"), "Shyvana": ("JGL",),
    "Singed": ("TOP",), "Sion": ("TOP",), "Sivir": ("ADC",), "Skarner": ("JGL", "TOP"),
    "Smolder": ("ADC", "MID", "TOP"), "Sona": ("SUP",), "Soraka": ("SUP",), "Swain": ("SUP", "MID", "ADC"),
    "Sylas": ("MID", "JGL", "TOP"), "Syndra": ("MID",), "TahmKench": ("TOP", "SUP"),
    "Taliyah": ("JGL", "MID"), "Talon": ("MID", "JGL"), "Taric": ("SUP",), "Teemo": ("TOP",),
    "Thresh": ("SUP",), "Tristana": ("ADC", "MID"), "Trundle": ("JGL", "TOP"), "Tryndamere": ("TOP",),
    "TwistedFate": ("MID",), "Twitch": ("ADC",), "Udyr": ("JGL", "TOP"), "Urgot": ("TOP",),
    "Varus": ("ADC", "MID"), "Vayne": ("ADC", "TOP"), "Veigar": ("MID",), "Velkoz": ("SUP", "MID"),
    "Vex": ("MID",), "Vi": ("JGL",), "Viego": ("JGL",), "Viktor": ("MID",), "Vladimir": ("MID", "TOP"),
    "Volibear": ("TOP", "JGL"), "Warwick": ("JGL", "TOP"), "Xayah": ("ADC",), "Xerath": ("SUP", "MID"),
    "XinZhao": ("JGL",), "Yasuo": ("MID", "TOP"), "Yone": ("MID", "TOP"), "Yorick": ("TOP",),
    "Yunara": ("ADC",), "Yuumi": ("SUP",), "Zaahen": ("TOP", "JGL"), "Zac": ("JGL", "TOP", "SUP"),
    "Zed": ("MID",), "Zeri": ("ADC",), "Ziggs": ("ADC", "MID"), "Zilean": ("SUP", "MID"),
    "Zoe": ("MID",), "Zyra": ("SUP", "JGL"),
}

# Archetypes that make no sense twice in the same team, even on different roles
# (e.g. Seraphine mid + Lulu sup = two enchanters).
ENCHANTERS = {
    "Janna", "Karma", "Lulu", "Milio", "Nami", "Renata", "Seraphine",
    "Sona", "Soraka", "Taric", "Yuumi",
}
ARCHETYPE_CAPS = [(ENCHANTERS, 1)]


def get_champion_roles(champion: str) -> Tuple[str, ...]:
    """All roles a champion can play (main role first). Unknown -> any role."""
    return CHAMPION_ROLES.get(champion, tuple(ROLES))


def get_champion_role(champion: str) -> str:
    """Main role (kept for backward compatibility)."""
    return get_champion_roles(champion)[0]


def find_assignment(team: List[str]) -> Optional[Dict[str, str]]:
    """
    Assign each picked champion to a distinct role it can play.
    Returns {champion: role} or None if impossible.
    Prefers main roles (lowest total role index).
    """
    team = [c for c in team if c]
    if len(team) > 5:
        return None
    best, best_cost = None, None
    for roles in permutations(ROLES, len(team)):
        cost = 0
        ok = True
        for champ, role in zip(team, roles):
            playable = get_champion_roles(champ)
            if role not in playable:
                ok = False
                break
            cost += playable.index(role)
        if ok and (best_cost is None or cost < best_cost):
            best, best_cost = dict(zip(team, roles)), cost
    return best


def respects_archetype_caps(team: List[str]) -> bool:
    return all(sum(c in group for c in team) <= cap for group, cap in ARCHETYPE_CAPS)


def is_valid_addition(team: List[str], candidate: str) -> bool:
    """True if candidate can join team while keeping a coherent composition."""
    new_team = [c for c in team if c] + [candidate]
    return respects_archetype_caps(new_team) and find_assignment(new_team) is not None


def role_for_candidate(team: List[str], candidate: str) -> Optional[str]:
    """Role the candidate would take in the best assignment."""
    assignment = find_assignment([c for c in team if c] + [candidate])
    return assignment.get(candidate) if assignment else None


def get_open_roles(team: List[str]) -> List[str]:
    """Roles that a new pick could still fill (accounts for flex picks)."""
    team = [c for c in team if c]
    open_roles = set()
    for roles in permutations(ROLES, len(team)):
        if all(r in get_champion_roles(c) for c, r in zip(team, roles)):
            open_roles |= set(ROLES) - set(roles)
    return [r for r in ROLES if r in open_roles]


# ---- backward-compatible helpers used by older scripts/tests ----

def get_missing_roles(team: List[str]) -> List[str]:
    return get_open_roles(team)


def get_role_count(team: List[str]) -> Dict[str, int]:
    count = {r: 0 for r in ROLES}
    assignment = find_assignment(team) or {}
    for role in assignment.values():
        count[role] += 1
    return count


def is_team_complete(team: List[str]) -> bool:
    return len([c for c in team if c]) == 5 and find_assignment(team) is not None


def get_role_priority_weight(champion: str, team: List[str]) -> float:
    return 1.0 if is_valid_addition(team, champion) else 0.0
