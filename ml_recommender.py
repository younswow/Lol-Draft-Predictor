

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

try:
    from model_loader import load_models, validate_models
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("model_loader not available")

try:
    from champion_roles import (
        get_champion_role, is_valid_addition, role_for_candidate,
    )
    ROLES_AVAILABLE = True
except ImportError:
    ROLES_AVAILABLE = False
    print("champion_roles not available")


@dataclass
class MLRecommendation:
    champion: str
    win_probability: float
    rank: int
    role: Optional[str] = None


class MLDraftRecommender:
    # ml draft recommender using trained v7 models.
    
    
    def __init__(self):
        self.models = None
        self.classifier = None
        self.embeddings_win = None
        self.embeddings_pop = None
        self.all_champions = None
        self.champ_to_idx = None
        self.win_rate = None
        self._load_models()
    
    def _load_models(self):
        if not MODELS_AVAILABLE:
            print("Model loader not available, using fallback")
            return
        
        try:
            self.models = load_models()
            validate_models(self.models)
            
            self.classifier = self.models['classifier']
            if hasattr(self.classifier, 'n_jobs'):
                self.classifier.n_jobs = 1  # one small batch per request: no multiprocess overhead
            self.embeddings_win = self.models['emb_win']
            self.embeddings_pop = self.models['emb_pop']
            self.all_champions = self.models.get('all_champions', [])
            self.champ_to_idx = self.models.get('lfm_item2id', {})
            self.win_rate = self.models.get('win_rate', np.array([]))
            
            print(f"ML Recommender ready ({len(self.all_champions)} champions)")
            
        except Exception as e:
            print(f"Failed to load models: {e}")
    
    def _extract_features(self, blue_team: List[str], red_team: List[str]) -> Optional[np.ndarray]:
        """Feature vector for a (blue, red) draft, same layout as training:
        [wr_blue, wr_red, wr_diff, syn_blue, syn_red, syn_diff, presence(+1 blue / -1 red)]."""
        if self.classifier is None:
            return None

        def wr(team):
            vals = []
            for c in team:
                idx = self.champ_to_idx.get(c)
                vals.append(float(self.win_rate[idx]) if idx is not None and idx < len(self.win_rate) else 0.5)
            return float(np.mean(vals)) if vals else 0.5

        def syn(team):
            vals = []
            for a, b in combinations(team, 2):
                ia, ib = self.champ_to_idx.get(a), self.champ_to_idx.get(b)
                if ia is None or ib is None:
                    continue  # unknown champion: no fake embedding
                vals.append(float(np.dot(self.embeddings_win[ia], self.embeddings_win[ib])))
            return float(np.mean(vals)) if vals else 0.0

        wb, wrd = wr(blue_team), wr(red_team)
        feat = [wb, wrd, wb - wrd]
        if self.embeddings_win is not None:
            sb, sr = syn(blue_team), syn(red_team)
            feat += [sb, sr, sb - sr]
        else:
            feat += [0.0, 0.0, 0.0]

        presence = [0.0] * len(self.all_champions)
        for c in blue_team:
            idx = self.champ_to_idx.get(c)
            if idx is not None:
                presence[idx] = 1.0
        for c in red_team:
            idx = self.champ_to_idx.get(c)
            if idx is not None:
                presence[idx] = -1.0
        feat += presence
        return np.array(feat, dtype=np.float32)

    def _calculate_counter_advantage(self, candidate: str, enemy_team: List[str]) -> float:
        #how candidate counter enemy team
        if not enemy_team or self.embeddings_win is None:
            return 0.5
        
        matchup_scores = []
        candidate_idx = self.champ_to_idx.get(candidate)
        
        if candidate_idx is None:
            return 0.5
        
        cand_emb = self.embeddings_win[candidate_idx]
        
        for enemy in enemy_team:
            enemy_idx = self.champ_to_idx.get(enemy)
            if enemy_idx is not None:
                enemy_emb = self.embeddings_win[enemy_idx]
                # dot product : positive = good matchup!
                matchup = np.dot(cand_emb, enemy_emb)
                matchup_score = (matchup + 1) / 2
                matchup_scores.append(np.clip(matchup_score, 0.2, 0.8))
        
        if not matchup_scores:
            return 0.5
        
        return np.mean(matchup_scores)
    
    def _calculate_team_synergy(self, candidate: str, ally_team: List[str]) -> float:
       #same, how candidate synergize with current team
        if not ally_team or self.embeddings_win is None:
            return 0.5
        
        synergy_scores = []
        candidate_idx = self.champ_to_idx.get(candidate)
        
        if candidate_idx is None:
            return 0.5
        
        cand_emb = self.embeddings_win[candidate_idx]
        
        for ally in ally_team:
            ally_idx = self.champ_to_idx.get(ally)
            if ally_idx is not None:
                ally_emb = self.embeddings_win[ally_idx]
                synergy = np.dot(cand_emb, ally_emb)
                synergy_score = (synergy + 1) / 2
                synergy_scores.append(np.clip(synergy_score, 0.2, 0.8))
        
        if not synergy_scores:
            return 0.5
        
        return np.mean(synergy_scores)
    
    def recommend(self, blue_team: List[str], red_team: List[str], available: List[str],
                  top_k: int = 3, side: str = "blue") -> List[MLRecommendation]:
        """Top-k picks for `side` ("blue" or "red").

        Hard rule: a candidate is kept only if the team stays coherent
        (every pick on a distinct role it can play, max one enchanter).
        """
        ally, enemy = (blue_team, red_team) if side == "blue" else (red_team, blue_team)
        ally = [c for c in ally if c]
        enemy = [c for c in enemy if c]

        if ROLES_AVAILABLE:
            available = [c for c in available if is_valid_addition(ally, c)]
        if not available:
            return []

        if self.classifier is None:
            return [MLRecommendation(c, 0.5, i + 1, role_for_candidate(ally, c) if ROLES_AVAILABLE else None)
                    for i, c in enumerate(available[:top_k])]

        # one batched prediction for all candidates (much faster than one call each)
        rows = []
        for cand in available:
            if side == "blue":
                rows.append(self._extract_features(blue_team + [cand], red_team))
            else:
                rows.append(self._extract_features(blue_team, red_team + [cand]))
        X = np.vstack(rows)
        p_blue = self.classifier.predict_proba(X)[:, 1]
        base = p_blue if side == "blue" else 1.0 - p_blue

        results = []
        for cand, base_prob in zip(available, base):
            counter_score = self._calculate_counter_advantage(cand, enemy)
            synergy_score = self._calculate_team_synergy(cand, ally)
            # ML prediction 60%, counter 20%, synergy 15%, constant role bonus 5% (role is now a hard filter)
            score = base_prob * 0.60 + counter_score * 0.20 + synergy_score * 0.15 + 0.05
            results.append((cand, float(score)))

        results.sort(key=lambda x: -x[1])
        return [
            MLRecommendation(c, p, i + 1, role_for_candidate(ally, c) if ROLES_AVAILABLE else None)
            for i, (c, p) in enumerate(results[:top_k])
        ]


def recommendations_to_api_format(recs: List[MLRecommendation]) -> List[Dict]:
    return [
        {
            "name": rec.champion,
            "pct": int(round(rec.win_probability * 100)),
            "role": rec.role,
        }
        for rec in recs
    ]


# Test
if __name__ == "__main__":
    rec = MLDraftRecommender()
    results = rec.recommend(
        blue_team=["Malphite", "Vi"],
        red_team=["Jayce"],
        available=["Orianna", "Thresh", "Lux", "Annie"],
        top_k=3
    )
    
    print("\nTest recommendations:")
    for r in results:
        print(f"  {r.rank}. {r.champion}: {r.win_probability:.1%}")
    
    print("\nAPI format:")
    api = recommendations_to_api_format(results)
    print(api)
