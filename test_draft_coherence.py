"""
Draft coherence tests: run with  python test_draft_coherence.py

Checks through the real Flask API that
  1. once roles are covered, only the missing role is recommended
  2. recommendations target the team whose turn it is (blue AND red)
  3. no two enchanters, no impossible compositions
  4. full drafts that always take the top suggestion end with 2 coherent teams
  5. latency per request
"""
import io
import random
import sys
import time
import contextlib

sys.path.insert(0, ".")
with contextlib.redirect_stdout(io.StringIO()):
    import app as A
from champion_roles import find_assignment, get_champion_roles, ENCHANTERS

client = A.app.test_client()
ORDER = A.DRAFT_ORDER
failures = []


def ask(step, bp, rp, bb=(), rb=()):
    with contextlib.redirect_stdout(io.StringIO()):
        r = client.post("/api/recommend", json={
            "step": step,
            "draft": {"blue": {"picks": list(bp), "bans": list(bb)},
                      "red": {"picks": list(rp), "bans": list(rb)}},
        })
    return r.get_json()


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# 1. last blue pick, only SUP missing (Yone is a mid, not a support)
j = ask(17, ["Ornn", "Sejuani", "Yone", "Kaisa"], ["Gnar", "Vi", "Azir"])
recs = [(r["name"], r["role"]) for r in j["recommendations"]]
check("Ornn/Sejuani/Yone/Kaisa -> only supports", all("SUP" in get_champion_roles(n) for n, _ in recs), recs)

# 1b. only ADC missing
j = ask(17, ["Ornn", "Graves", "Ahri", "Thresh"], [])
recs = [(r["name"], r["role"]) for r in j["recommendations"]]
check("Ornn/Graves/Ahri/Thresh -> only ADC", all(r == "ADC" for _, r in recs), recs)

# 1c. flex: Gragas can go top, so ADC is still the only open role
j = ask(17, ["Gragas", "Sejuani", "Orianna", "Thresh"], [])
recs = [(r["name"], r["role"]) for r in j["recommendations"]]
check("Gragas(flex)/Sejuani/Orianna/Thresh -> only ADC", all(r == "ADC" for _, r in recs), recs)

# 2. red's turn: recommendations must fit RED's team, not blue's
j = ask(18, ["Ornn", "Sejuani", "Ahri", "Kaisa", "Lulu"], ["Gnar", "Vi", "Azir"])
recs = [(r["name"], r["role"]) for r in j["recommendations"]]
check("Red pick 4 (red has TOP/JGL/MID) -> ADC or SUP", all(r in ("ADC", "SUP") for _, r in recs) and j["team"] == "red", recs)

# 3. no second enchanter
j = ask(16, ["Ornn", "LeeSin", "Lulu"], ["Gnar"])
names = [r["name"] for r in j["recommendations"]]
check("Lulu already picked -> no enchanter suggested", not (set(names) & ENCHANTERS), names)

# 3b. ban phase targets the enemy
j = ask(11, ["Ornn", "Sejuani", "Ahri"], ["Gnar", "Vi"])
recs = [(r["name"], r["role"]) for r in j["recommendations"]]
check("Red ban 4 -> threats for BLUE (ADC/SUP)", all(r in ("ADC", "SUP") for _, r in recs), recs)

# 4. full drafts, always taking top-1 at pick steps, random bans
pool = list(A.CHAMPIONS)
bad = 0
latencies = []
for seed in range(20):
    random.seed(seed)
    st = {"blue": {"picks": [], "bans": []}, "red": {"picks": [], "bans": []}}
    for step, (team, action) in enumerate(ORDER):
        used = set(sum([st[t][k] for t in st for k in st[t]], []))
        if action == "ban":
            st[team]["bans"].append(random.choice([c for c in pool if c not in used]))
            continue
        t0 = time.time()
        j = ask(step, st["blue"]["picks"], st["red"]["picks"], st["blue"]["bans"], st["red"]["bans"])
        latencies.append(time.time() - t0)
        recs = j["recommendations"]
        # also simulate a human sometimes picking the 2nd/3rd suggestion
        st[team]["picks"].append(recs[min(len(recs) - 1, random.choice([0, 0, 1, 2]))]["name"])
    for t in ("blue", "red"):
        team = st[t]["picks"]
        ok = find_assignment(team) is not None and sum(c in ENCHANTERS for c in team) <= 1
        if not ok:
            bad += 1
        if seed < 3:
            print(f"   seed {seed} {t:4}: {find_assignment(team)}")
check("20 full drafts -> 40 coherent teams", bad == 0, f"{bad} incoherent")

latencies.sort()
p50 = latencies[len(latencies) // 2] * 1000
p95 = latencies[int(len(latencies) * 0.95)] * 1000
print(f"Latency per request: median {p50:.0f} ms, p95 {p95:.0f} ms")

print("\nALL GOOD" if not failures else f"\n{len(failures)} FAILURE(S): {failures}")
sys.exit(1 if failures else 0)
