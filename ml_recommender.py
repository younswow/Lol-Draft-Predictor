

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
        get_champion_role, get_picked_roles, get_available_by_role,
        get_missing_roles, get_role_priority_weight
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
            self.embeddings_win = self.models['emb_win']
            self.embeddings_pop = self.models['emb_pop']
            self.all_champions = self.models.get('all_champions', [])
            self.champ_to_idx = self.models.get('lfm_item2id', {})
            self.win_rate = self.models.get('win_rate', np.array([]))
            
            print(f"ML Recommender ready ({len(self.all_champions)} champions)")
            
        except Exception as e:
            print(f"Failed to load models: {e}")
    
    def _extract_features(self, blue_team: List[str], red_team: List[str], candidate: str) -> Optional[np.ndarray]:
       
        if not self.classifier:
            return None
        
     
        test_blue = blue_team + [candidate]
        test_red = red_team
        
        try:
           
            feat = []
            
            
            wr_blue = []
            for c in test_blue:
                idx = self.champ_to_idx.get(c)
                if idx is not None and idx < len(self.win_rate):
                    wr_blue.append(float(self.win_rate[idx]))
                else:
                    wr_blue.append(0.5)
            
            wr_red = []
            for c in test_red:
                idx = self.champ_to_idx.get(c)
                if idx is not None and idx < len(self.win_rate):
                    wr_red.append(float(self.win_rate[idx]))
                else:
                    wr_red.append(0.5)
            
            feat.append(np.mean(wr_blue))
            feat.append(np.mean(wr_red))
            feat.append(np.mean(wr_blue) - np.mean(wr_red))
            
           
            if self.embeddings_win is not None:
                syn_blue = []
                for a, b in combinations(test_blue, 2):
                    idx_a = self.champ_to_idx.get(a, 0)
                    idx_b = self.champ_to_idx.get(b, 0)
                    if idx_a < len(self.embeddings_win) and idx_b < len(self.embeddings_win):
                        syn = np.dot(
                            self.embeddings_win[idx_a],
                            self.embeddings_win[idx_b]
                        )
                        syn_blue.append(syn)
                    else:
                        syn_blue.append(0.0)
                
                syn_red = []
                for a, b in combinations(test_red, 2):
                    idx_a = self.champ_to_idx.get(a, 0)
                    idx_b = self.champ_to_idx.get(b, 0)
                    if idx_a < len(self.embeddings_win) and idx_b < len(self.embeddings_win):
                        syn = np.dot(
                            self.embeddings_win[idx_a],
                            self.embeddings_win[idx_b]
                        )
                        syn_red.append(syn)
                    else:
                        syn_red.append(0.0)
                
                if not syn_blue:
                    syn_blue = [0.0]
                if not syn_red:
                    syn_red = [0.0]
                
                feat.append(np.mean(syn_blue))
                feat.append(np.mean(syn_red))
                feat.append(np.mean(syn_blue) - np.mean(syn_red))
            else:
                feat.extend([0.0, 0.0, 0.0])
            
           
            if self.all_champions:
                presence = [0.0] * len(self.all_champions)
                for c in test_blue:
                    idx = self.champ_to_idx.get(c)
                    if idx is not None and idx < len(presence):
                        presence[idx] = 1.0
                for c in test_red:
                    idx = self.champ_to_idx.get(c)
                    if idx is not None and idx < len(presence):
                        presence[idx] = -1.0
                feat.extend(presence)
            
            return np.array([feat], dtype=np.float32)
        
        except Exception as e:
            return None
    
    def _calculate_counter_advantage(self, candidate: str, enemy_team: List[str]) -> float:
        #how candidate counter enemy team
        if not enemy_team or self.embeddings_win is None:
            return 0.5
        
        matchup_scores = []
        candidate_idx = self.champ_to_idx.get(candidate, 0)
        
        if candidate_idx >= len(self.embeddings_win):
            return 0.5
        
        cand_emb = self.embeddings_win[candidate_idx]
        
        for enemy in enemy_team:
            enemy_idx = self.champ_to_idx.get(enemy, 0)
            if enemy_idx < len(self.embeddings_win):
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
        candidate_idx = self.champ_to_idx.get(candidate, 0)
        
        if candidate_idx >= len(self.embeddings_win):
            return 0.5
        
        cand_emb = self.embeddings_win[candidate_idx]
        
        for ally in ally_team:
            ally_idx = self.champ_to_idx.get(ally, 0)
            if ally_idx < len(self.embeddings_win):
                ally_emb = self.embeddings_win[ally_idx]
                synergy = np.dot(cand_emb, ally_emb)
                synergy_score = (synergy + 1) / 2
                synergy_scores.append(np.clip(synergy_score, 0.2, 0.8))
        
        if not synergy_scores:
            return 0.5
        
        return np.mean(synergy_scores)
    
    def recommend(self, blue_team: List[str], red_team: List[str], available: List[str], top_k: int = 3) -> List[MLRecommendation]:
       #generate top k recommendations
        print(f"DEBUG: recommend() called with {len(available)} available champs")
        print(f"DEBUG: blue_team={blue_team}, missing ROLES_AVAILABLE={ROLES_AVAILABLE}")
        
        if self.classifier is None:
            # fallback returns first k available
            return [
                MLRecommendation(champ, 0.5, i+1)
                for i, champ in enumerate(available[:top_k])
            ]
        
        # role filtering 
        if ROLES_AVAILABLE:
            from champion_roles import get_missing_roles
            missing_roles = get_missing_roles(blue_team)
            print(f"DEBUG: missing_roles={missing_roles}")
            
            if missing_roles:
                candidates_by_missing_roles = [c for c in available if get_champion_role(c) in missing_roles]
                print(f"DEBUG: filtered from {len(available)} to {len(candidates_by_missing_roles)} by role")
                print(f"DEBUG: candidates_by_missing_roles={candidates_by_missing_roles}")
                if candidates_by_missing_roles:
                    available = candidates_by_missing_roles
        
        results = []
        
        for candidate in available:
            try:
                
                X = self._extract_features(blue_team, red_team, candidate)
                if X is None or X.shape[1] == 0:
                    base_prob = 0.5
                else:
                    
                    if hasattr(self.classifier, 'predict_proba'):
                        y_pred = self.classifier.predict_proba(X)
                        if y_pred.shape[1] > 1:
                            base_prob = y_pred[0][1]
                        else:
                            base_prob = 0.5
                    elif hasattr(self.classifier, 'decision_function'):
                        score = self.classifier.decision_function(X)[0]
                        from scipy.special import expit
                        base_prob = expit(score)
                    else:
                        base_prob = 0.5
                
                
                counter_score = self._calculate_counter_advantage(candidate, red_team)
                synergy_score = self._calculate_team_synergy(candidate, blue_team)
                
                
                role_weight = 1.0
                if ROLES_AVAILABLE:
                    role_weight = get_role_priority_weight(candidate, blue_team)
                
                # combine scores ML prediction: 60%, counter-pick: 20%, synergy: 15%, role: 5%
                weighted_prob = (
                    base_prob * 0.60 +
                    counter_score * 0.20 +
                    synergy_score * 0.15 +
                    (role_weight / 1.5) * 0.05  # normalize
                )
                
                results.append((candidate, float(weighted_prob), role_weight))
            
            except Exception as e:
                results.append((candidate, 0.5, 1.0))
        
        
        results.sort(key=lambda x: -x[1])
        
        
        return [
            MLRecommendation(champ, prob, i+1)
            for i, (champ, prob, _) in enumerate(results[:top_k])
        ]


def recommendations_to_api_format(recs: List[MLRecommendation]) -> List[Dict]:
    return [
        {
            "name": rec.champion,
            "pct": int(rec.win_probability * 100)
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
