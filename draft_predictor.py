
#draft predictor

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import combinations
import warnings
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# data structure

@dataclass
class DraftState: # current state of draft
    blue_picks: List[str]
    blue_bans: List[str]
    red_picks: List[str]
    red_bans: List[str]
    
    def get_used_champions(self) -> Set[str]:
        return set(self.blue_picks + self.blue_bans + self.red_picks + self.red_bans)
    
    def get_available_champions(self, all_champions: List[str]) -> List[str]:
        used = self.get_used_champions()
        return [c for c in all_champions if c not in used]
    
    def blue_team_complete(self) -> bool:
        return len(self.blue_picks) == 5
    
    def red_team_complete(self) -> bool:
        return len(self.red_picks) == 5
    
    def copy(self):
        return DraftState(
            blue_picks=self.blue_picks.copy(),
            blue_bans=self.blue_bans.copy(),
            red_picks=self.red_picks.copy(),
            red_bans=self.red_bans.copy()
        )


@dataclass
class RecommendationResult:
    champion: str
    win_probability: float  
    rank: int               
    team_score: float       # team strength after pick
    synergy_avg: float      
    counter_score: float    # score against enemy team



class DraftEncoder:
    
    
    def __init__(self, feature_engineering):
       
        self.fe = feature_engineering
        self.all_champs = sorted(feature_engineering.all_champions)
        self.n_champs = len(self.all_champs)
    
    def encode_partial_team(self, team_champs: List[str]) -> Dict:
        
        features = {}
        
        # Synergies among existing picks
        synergies = []
        for a, b in combinations(team_champs, 2):
            synergies.append(self.fe.synergy_wr(a, b))
        
        features['syn_count'] = len(synergies)
        features['syn_mean'] = np.mean(synergies) if synergies else 0.0
        features['syn_min'] = np.min(synergies) if synergies else 0.0
        features['syn_max'] = np.max(synergies) if synergies else 0.0
        
        # Individual win rates
        wrs = [self.fe.champ_wr(c) for c in team_champs]
        features['wr_mean'] = np.mean(wrs) if wrs else 0.5
        features['wr_min'] = np.min(wrs) if wrs else 0.5
        features['wr_max'] = np.max(wrs) if wrs else 0.5
        
        # LightFM synergies
        fm_scores = []
        for a, b in combinations(team_champs, 2):
            fm_scores.append(self.fe.fm_synergy(a, b))
        features['fm_mean'] = np.mean(fm_scores) if fm_scores else 0.0
        features['fm_min'] = np.min(fm_scores) if fm_scores else 0.0
        features['fm_max'] = np.max(fm_scores) if fm_scores else 0.0
        
        # Uplift synergies (corrected for popularity)
        uplift_scores = []
        for a, b in combinations(team_champs, 2):
            uplift_scores.append(self.fe.uplift_synergy(a, b))
        features['uplift_mean'] = np.mean(uplift_scores) if uplift_scores else 0.0
        features['uplift_min'] = np.min(uplift_scores) if uplift_scores else 0.0
        features['uplift_max'] = np.max(uplift_scores) if uplift_scores else 0.0
        
        # Macro complementarity
        macro_scores = []
        for a, b in combinations(team_champs, 2):
            macro_scores.append(self.fe.macro_comp(a, b))
        features['macro_mean'] = np.mean(macro_scores) if macro_scores else 0.0
        features['macro_min'] = np.min(macro_scores) if macro_scores else 0.0
        features['macro_max'] = np.max(macro_scores) if macro_scores else 0.0
        
        # Team size (how complete is the team)
        features['team_size'] = len(team_champs) / 5.0
        
        return features
    
    def encode_matchup(self, team_picks: List[str], enemy_picks: List[str]) -> Dict:
        
        features = {}
        
        if not team_picks or not enemy_picks:
            features['counter_mean'] = 0.0
            features['counter_min'] = 0.0
            features['counter_max'] = 0.0
            return features
        
        counters = []
        for attacker in team_picks:
            for defender in enemy_picks:
                counters.append(self.fe.counter_wr(attacker, defender))
        
        features['counter_mean'] = np.mean(counters) if counters else 0.5
        features['counter_min'] = np.min(counters) if counters else 0.5
        features['counter_max'] = np.max(counters) if counters else 0.5
        
        # nb of favorable matchups (WR > 0.5)
        favorable = sum(1 for c in counters if c > 0.5)
        features['favorable_matchups'] = favorable / len(counters) if counters else 0.0
        
        return features
    
    def encode_candidate_synergy(self, candidate: str, team_champs: List[str]) -> Dict:
    
        features = {}
        
        if not team_champs:
            features['candidate_syn'] = 0.0
            features['candidate_uplift'] = 0.0
            features['candidate_macro'] = 0.0
            features['candidate_wr'] = self.fe.champ_wr(candidate)
            return features
        
        synergies = [self.fe.synergy_wr(candidate, c) for c in team_champs]
        uplifts = [self.fe.uplift_synergy(candidate, c) for c in team_champs]
        macros = [self.fe.macro_comp(candidate, c) for c in team_champs]
        
        features['candidate_syn'] = np.mean(synergies)
        features['candidate_uplift'] = np.mean(uplifts)
        features['candidate_macro'] = np.mean(macros)
        features['candidate_wr'] = self.fe.champ_wr(candidate)
        
        return features
    
    def encode_counter_matchup(self, candidate: str, enemy_picks: List[str]) -> Dict:
        # how much THE candidate counter the enemy team
        features = {}
        
        if not enemy_picks:
            features['candidate_vs_enemy'] = 0.5
            features['hard_countered'] = 0.0
            features['hard_counter_enemy'] = 0.0
            return features
        
        matchups = [self.fe.counter_wr(candidate, enemy) for enemy in enemy_picks]
        features['candidate_vs_enemy'] = np.mean(matchups)
        features['hard_countered'] = sum(1 for m in matchups if m < 0.4) / len(matchups)
        features['hard_counter_enemy'] = sum(1 for m in matchups if m > 0.6) / len(matchups)
        
        return features
    
    def full_encode(self, draft_state: DraftState, candidate: str) -> np.ndarray:
       
        features = {}
        
        # 1 encode allied team (with candidate added)
        blue_with_candidate = draft_state.blue_picks + [candidate]
        blue_features = self.encode_partial_team(blue_with_candidate)
        features.update({f'blue_{k}': v for k, v in blue_features.items()})
        
        # 2 encode enemy team
        red_features = self.encode_partial_team(draft_state.red_picks)
        features.update({f'red_{k}': v for k, v in red_features.items()})
        
        # 3 encode matchup (blue vs red)
        matchup_features = self.encode_matchup(blue_with_candidate, draft_state.red_picks)
        features.update({f'matchup_{k}': v for k, v in matchup_features.items()})
        
        # 4 encode candidate contribution
        candidate_syn = self.encode_candidate_synergy(candidate, draft_state.blue_picks)
        features.update({f'candidate_{k}': v for k, v in candidate_syn.items()})
        
        # 5 encode counter matchup
        counter_features = self.encode_counter_matchup(candidate, draft_state.red_picks)
        features.update({f'counter_{k}': v for k, v in counter_features.items()})
        
        
        feat_vector = np.array([
            features.get(f'blue_syn_mean', 0.0),
            features.get(f'blue_syn_min', 0.0),
            features.get(f'blue_syn_max', 0.0),
            features.get(f'red_syn_mean', 0.0),
            features.get(f'red_syn_min', 0.0),
            features.get(f'red_syn_max', 0.0),
            features.get(f'blue_syn_mean', 0.0) - features.get(f'red_syn_mean', 0.0),
            features.get(f'matchup_counter_mean', 0.5),
            features.get(f'matchup_counter_min', 0.5),
            features.get(f'matchup_counter_max', 0.5),
            features.get(f'blue_wr_mean', 0.5),
            features.get(f'red_wr_mean', 0.5),
            features.get(f'blue_wr_mean', 0.5) - features.get(f'red_wr_mean', 0.5),
            features.get(f'blue_fm_mean', 0.0),
            features.get(f'red_fm_mean', 0.0),
            features.get(f'blue_fm_mean', 0.0) - features.get(f'red_fm_mean', 0.0),
            features.get(f'blue_uplift_mean', 0.0),
            features.get(f'red_uplift_mean', 0.0),
            features.get(f'blue_uplift_mean', 0.0) - features.get(f'red_uplift_mean', 0.0),
            features.get(f'blue_macro_mean', 0.0),
            features.get(f'red_macro_mean', 0.0),
            features.get(f'blue_macro_mean', 0.0) - features.get(f'red_macro_mean', 0.0),
        ], dtype=np.float32)
        
        return feat_vector, features




class DraftTeamEvaluator:
    
    def __init__(self, feature_engineering, trained_model):
        
        self.fe = feature_engineering
        self.model = trained_model
        self.encoder = DraftEncoder(feature_engineering)
    
    def evaluate_team(self, blue_team: List[str], red_team: List[str]) -> Dict:
      
        if len(blue_team) != 5 or len(red_team) != 5:
            return {
                'win_prob': 0.5,
                'blue_score': 0.0,
                'red_score': 0.0,
                'matchup_delta': 0.0
            }
        
        # pseudo draft state
        draft_state = DraftState(
            blue_picks=blue_team,
            blue_bans=[],
            red_picks=red_team,
            red_bans=[]
        )
        
        
        candidate = blue_team[-1]
        feat_vector, _ = self.encoder.full_encode(draft_state, candidate)
        
        # get prediction
        if hasattr(self.model, 'predict_proba'):
            win_prob = float(self.model.predict_proba(feat_vector.reshape(1, -1))[0, 1])
        else:
            # fallback for pipeline
            win_prob = float(self.model.predict_proba(feat_vector.reshape(1, -1))[0, 1])
        
        #1.0 = 100% win probability, 0.0 = 0% win probability
        blue_score = win_prob
        red_score = 1.0 - win_prob
        matchup_delta = blue_score - red_score
        
        return {
            'win_prob': win_prob,
            'blue_score': blue_score,
            'red_score': red_score,
            'matchup_delta': matchup_delta
        }
    
    def score_candidate_impact(self, draft_state: DraftState, candidate: str) -> float:
    
        if len(draft_state.blue_picks) >= 5:
            return 0.5  
        
        # hypothetical team with candidate
        hypothetical_blue = draft_state.blue_picks + [candidate]
        
        # if both teams incomplete, we pad with placeholders for evaluation
        red_team = draft_state.red_picks + ['placeholder'] * (5 - len(draft_state.red_picks))
        
        if len(hypothetical_blue) == 5 and len(red_team) == 5:
            result = self.evaluate_team(hypothetical_blue, red_team)
            return result['win_prob']
        
        # fallback: estimate based on candidate synergy + counter + win rate
        features = {}
        features['syn'] = np.mean([
            self.fe.synergy_wr(candidate, c) for c in draft_state.blue_picks
        ]) if draft_state.blue_picks else 0.0
        features['uplift'] = np.mean([
            self.fe.uplift_synergy(candidate, c) for c in draft_state.blue_picks
        ]) if draft_state.blue_picks else 0.0
        features['macro'] = np.mean([
            self.fe.macro_comp(candidate, c) for c in draft_state.blue_picks
        ]) if draft_state.blue_picks else 0.0
        features['wr'] = self.fe.champ_wr(candidate)
        features['counter'] = np.mean([
            self.fe.counter_wr(candidate, c) for c in draft_state.red_picks
        ]) if draft_state.red_picks else 0.5
        
        # weighted combination
        score = (
            0.2 * features['syn'] +        
            0.3 * features['uplift'] +     
            0.15 * features['macro'] +     
            0.25 * features['wr'] +        
            0.1 * features['counter']      
        )
        
        
        win_prob = 0.5 + score * 0.2  
        return np.clip(win_prob, 0.3, 0.7)




class DraftRecommender:
   
    
    def __init__(self, feature_engineering, trained_model, all_champions: List[str]):
      
        self.fe = feature_engineering
        self.model = trained_model
        self.evaluator = DraftTeamEvaluator(feature_engineering, trained_model)
        self.all_champions = sorted(all_champions)
    
    def recommend(self, draft_state: DraftState, top_k: int = 3) -> List[RecommendationResult]:
       #top k reco generation
        available = draft_state.get_available_champions(self.all_champions)
        
        if not available:
            return []
        
        scores = []
        
        for candidate in available:
            # 1 predict win probability with this candidate
            win_prob = self.evaluator.score_candidate_impact(draft_state, candidate)
            
            # 2 calculate synergy contribution
            synergy_scores = [
                self.fe.synergy_wr(candidate, c) for c in draft_state.blue_picks
            ]
            synergy_avg = np.mean(synergy_scores) if synergy_scores else 0.0
            
            # 3 calculate counter value
            counter_scores = [
                self.fe.counter_wr(candidate, c) for c in draft_state.red_picks
            ]
            counter_score = np.mean(counter_scores) if counter_scores else 0.5
            
            # 4 team score (simplified, for ranking)
            team_score = np.mean([
                self.fe.champ_wr(candidate),
                synergy_avg,
                counter_score
            ])
            
            scores.append({
                'champion': candidate,
                'win_prob': win_prob,
                'synergy_avg': synergy_avg,
                'counter_score': counter_score,
                'team_score': team_score
            })
        
        # sort by win probability (descending)
        scores.sort(key=lambda x: x['win_prob'], reverse=True)
        
        results = []
        for rank, score in enumerate(scores[:top_k], 1):
            results.append(RecommendationResult(
                champion=score['champion'],
                win_probability=score['win_prob'],
                rank=rank,
                team_score=score['team_score'],
                synergy_avg=score['synergy_avg'],
                counter_score=score['counter_score']
            ))
        
        return results
    
    def batch_evaluate(self, draft_state: DraftState, champions: List[str]) -> pd.DataFrame:
        #we evaluate a batch of multiple candidates 
        results = []
        for champion in champions:
            win_prob = self.evaluator.score_candidate_impact(draft_state, champion)
            synergy_scores = [
                self.fe.synergy_wr(champion, c) for c in draft_state.blue_picks
            ]
            synergy_avg = np.mean(synergy_scores) if synergy_scores else 0.0
            counter_scores = [
                self.fe.counter_wr(champion, c) for c in draft_state.red_picks
            ]
            counter_score = np.mean(counter_scores) if counter_scores else 0.5
            
            results.append({
                'champion': champion,
                'win_probability': win_prob,
                'synergy_avg': synergy_avg,
                'counter_score': counter_score
            })
        
        return pd.DataFrame(results).sort_values('win_probability', ascending=False)



def parse_draft_from_ui(draft_json: Dict) -> DraftState:
   #UI
    blue = draft_json.get('blue', {})
    red = draft_json.get('red', {})
    
    return DraftState(
        blue_picks=blue.get('picks', []),
        blue_bans=blue.get('bans', []),
        red_picks=red.get('picks', []),
        red_bans=red.get('bans', [])
    )


def recommendations_to_json(recommendations: List[RecommendationResult]) -> List[Dict]:
    #for UI
    return [
        {
            'name': r.champion,
            'pct': int(r.win_probability * 100),
            'rank': r.rank,
            'synergy': round(r.synergy_avg, 3),
            'counter': round(r.counter_score, 3)
        }
        for r in recommendations
    ]
