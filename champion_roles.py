"""
Champion role detection and analysis
"""

# champion to primary role mapping (important for filtering!) (based on LoL meta)
CHAMPION_ROLES = {
    # TOP
    "Aatrox": "TOP", "Ambessa": "TOP", "Camille": "TOP", "Chogath": "TOP",
    "Darius": "TOP", "Diana": "TOP", "DrMundo": "TOP", "Fiora": "TOP",
    "Garen": "TOP", "Gnar": "TOP", "Gragas": "TOP", "Gwen": "TOP",
    "Hecarim": "TOP", "Illaoi": "TOP", "Irelia": "TOP", "Jax": "TOP",
    "Jayce": "TOP", "Jhin": "TOP", "Kayle": "TOP", "Kled": "TOP",
    "Malphite": "TOP", "Maokai": "TOP", "Mordekaiser": "TOP", "Ornn": "TOP",
    "Pantheon": "TOP", "Quinn": "TOP", "Rammus": "TOP", "Renekton": "TOP",
    "Rengar": "TOP", "Riven": "TOP", "Rumble": "TOP", "Shen": "TOP",
    "Shyvana": "TOP", "Singed": "TOP", "Sion": "TOP", "Swain": "TOP",
    "Tahm Kench": "TOP", "TahmKench": "TOP", "Teemo": "TOP", "Trundle": "TOP",
    "Tryndamere": "TOP", "Udyr": "TOP", "Urgot": "TOP", "Vladimir": "TOP",
    "Volibear": "TOP", "Warwick": "TOP", "Wukong": "TOP", "Yasuo": "TOP",
    "Yorick": "TOP",
    
    # JGL
    "Amumu": "JGL", "Belveth": "JGL", "Ekko": "JGL", "Elise": "JGL",
    "Evelynn": "JGL", "FiddleSticks": "JGL", "Graves": "JGL", "Hecarim": "JGL",
    "Ivern": "JGL", "Jarvan IV": "JGL", "JarvanIV": "JGL", "Karthus": "JGL",
    "Khazix": "JGL", "Kindred": "JGL", "LeeSin": "JGL", "Lillia": "JGL",
    "MonkeyKing": "JGL", "Nidalee": "JGL", "Nocturne": "JGL", "Nunu": "JGL",
    "Olaf": "JGL", "Rammus": "JGL", "RekSai": "JGL", "Rengar": "JGL",
    "Sejuani": "JGL", "Shaco": "JGL", "Shyvana": "JGL", "Sylas": "JGL",
    "Taliyah": "JGL", "Trundle": "JGL", "Udyr": "JGL", "Vi": "JGL",
    "Warwick": "JGL", "XinZhao": "JGL", "Zac": "JGL",
    
    # MID
    "Ahri": "MID", "Akali": "MID", "Akshan": "MID", "AurelionSol": "MID",
    "Azir": "MID", "Brand": "MID", "Cassiopeia": "MID", "Diana": "MID",
    "Ekko": "MID", "Evelynn": "MID", "Ezreal": "MID", "Fizz": "MID",
    "Galio": "MID", "Gragas": "MID", "Hwei": "MID", "Irelia": "MID",
    "Jayce": "MID", "Jhin": "MID", "Kassadin": "MID", "Katarina": "MID",
    "Kayle": "MID", "Kayn": "MID", "Leblanc": "MID", "Lissandra": "MID",
    "Lux": "MID", "Malzahar": "MID", "Mordekaiser": "MID", "Neeko": "MID",
    "Orianna": "MID", "Pantheon": "MID", "Pyke": "MID", "Qiyana": "MID",
    "Ryze": "MID", "Seraphine": "MID", "Swain": "MID", "Sylas": "MID",
    "Syndra": "MID", "Taliyah": "MID", "Talon": "MID", "TwistedFate": "MID",
    "Veigar": "MID", "Velkoz": "MID", "Vex": "MID", "Viktor": "MID",
    "Vladimir": "MID", "Xerath": "MID", "Yasuo": "MID", "Zed": "MID",
    "Ziggs": "MID", "Zilean": "MID", "Zoe": "MID",
    
    # ADC
    "Aphelios": "ADC", "Ashe": "ADC", "Caitlyn": "ADC", "Corki": "ADC",
    "Draven": "ADC", "Ezreal": "ADC", "Jhin": "ADC", "Jinx": "ADC",
    "Kaisa": "ADC", "Kalista": "ADC", "Kennen": "ADC", "KogMaw": "ADC",
    "Lucian": "ADC", "MissFortune": "ADC", "Nilah": "ADC", "Pyke": "ADC",
    "Quinn": "ADC", "Samira": "ADC", "Sivir": "ADC", "Smolder": "ADC",
    "Teemo": "ADC", "Tristana": "ADC", "Twitch": "ADC", "Urgot": "ADC",
    "Varus": "ADC", "Vayne": "ADC", "Xayah": "ADC",
    
    # SUP (keeping only pure support champions, removing multi-role overlaps)
    "Alistar": "SUP", "Bard": "SUP", "Blitzcrank": "SUP", "Braum": "SUP",
    "Briar": "SUP", "Fiddlesticks": "SUP", "Heimerdinger": "SUP", "Janna": "SUP",
    "KSante": "SUP", "Leona": "SUP", "Lulu": "SUP", "Mel": "SUP", "Milio": "SUP",
    "Morgana": "SUP", "Nami": "SUP", "Nautilus": "SUP", "Rell": "SUP", "Renata": "SUP",
    "Rakan": "SUP", "Sona": "SUP", "Soraka": "SUP", "Taric": "SUP",
    "Thresh": "SUP", "Yuumi": "SUP",
}

def get_champion_role(champion_name: str) -> str:
    """Get the primary role of a champion"""
    role = CHAMPION_ROLES.get(champion_name)
    if role:
        return role
    

    name_lower = champion_name.lower()
    if any(x in name_lower for x in ['annie', 'brand', 'lux', 'ahri']):
        return 'MID'
    elif any(x in name_lower for x in ['jhin', 'ashe', 'jinx', 'adc']):
        return 'ADC'
    elif any(x in name_lower for x in ['garen', 'malphite', 'darius', 'top']):
        return 'TOP'
    elif any(x in name_lower for x in ['lee', 'graves', 'elise', 'jgl']):
        return 'JGL'
    else:
        return 'SUP'  

def get_picked_roles(team_picks: list) -> set:
    """Get the roles already picked in the team"""
    roles = set()
    for champ in team_picks:
        role = get_champion_role(champ)
        roles.add(role)
    return roles

def get_available_by_role(available: list) -> dict:
    """Organize available champions by role"""
    by_role = {}
    for champ in available:
        role = get_champion_role(champ)
        if role not in by_role:
            by_role[role] = []
        by_role[role].append(champ)
    return by_role

def get_role_count(team: list) -> dict:
    """Count how many champions of each role are in team"""
    count = {"TOP": 0, "JGL": 0, "MID": 0, "ADC": 0, "SUP": 0}
    for champ in team:
        role = get_champion_role(champ)
        if role in count:
            count[role] += 1
    return count

def get_missing_roles(team: list) -> list:
    """Get list of roles that still need to be picked"""
    count = get_role_count(team)
    missing = []
    
    # Ideal team: 1 TOP, 1 JGL, 1 MID, 1 ADC, 1 SUP
    for role in ["TOP", "JGL", "MID", "ADC", "SUP"]:
        if count.get(role, 0) < 1:
            missing.append(role)
    
    return missing

def is_team_complete(team: list) -> bool:
    """Check if team has all 5 different roles"""
    if len(team) != 5:
        return False
    roles = get_role_count(team)
    return all(roles.get(role, 0) >= 1 for role in ["TOP", "JGL", "MID", "ADC", "SUP"])

def get_role_priority_weight(champion: str, team: list) -> float:
    """
    get priority weight for picking this champion based on team needs.
    higher = more important to fill this role.
    
    weight score between 0.3 (duplicate role) and 1.5 (critical role)
    idea is prio missing roles but not completely
    ignore duplicate (sometimes you want to play APC syndra bot with AD mid for example) so 
    we need to allow it 
    """
    role = get_champion_role(champion)
    missing = get_missing_roles(team)
    count = get_role_count(team)
    
    # if team is complete, all equally important
    if len(missing) == 0:
        return 1.0
    
    # if this role is missing, it's critical
    if role in missing:
        return 1.5
    
    # if role is already picked, it's less important (duplicate)
    if count.get(role, 0) > 0:
        return 0.3  # penalize duplicate roles heavily !!
    
    return 1.0
