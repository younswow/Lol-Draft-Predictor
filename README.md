# LoL Draft Predictor v7

A champion recommendation system for League of Legends based on Machine Learning (LDA), matchup analysis, and team synergy evaluation.

Draft progression matters: Recommendations naturally improve as the draft advances, don't hesitate to priority-pick early on strong comfort/prio picks rather than waiting for a "perfect counter".

---

## Quick Start

### Install Dependencies

```bash
pip install numpy pandas scikit-learn flask scipy
```

You will need to have the dataset on your computer called `lol_dataset_challenger_grandmaster_clean.csv`.

### Train the Model

```bash
python train_v7_fast.py
```

### Launch the Server

```bash
python app.py
```

---

## Features

- **ML-Powered:** Trained on 14,000+ Master+ ranked games from EUW servers.
- **Matchups & Synergies:** Calculates counters against the enemy team and compatibility with allied picks.
- **Role-Aware System:** Detects missing roles in the current composition and prioritizes relevant champions (e.g., prioritizes Supports if the role is vacant).
- **Dynamic Updates:** Recalculates recommendations in real-time at every step of the draft.

---

## Scoring Logic

The final recommendation score is a weighted average of four factors (prioritizes tactical real time signals over passive global predictions) : 

| Factor | Weight | Description |
|---|---|---|
| ML Model | 60% | Raw win probability from the trained classifier. |
| Counter-pick | 20% | Statistical advantage against the enemy team's picks. |
| Synergy | 15% | Synergy with existing allied champions. |
| Role Weight | 5% | Bonus applied for filling a missing role in the team composition. |

---

## Project Structure

| File | Description |
|---|---|
| `app.py` | Flask server and API management. |
| `ml_recommender.py` | Main recommendation engine and scoring logic. |
| `champion_roles.py` | Role database and team composition analysis. |
| `model_loader.py` | Utility to load saved model files. |
| `train_v7_fast.py` | Training script for the LDA model. |
| `models/` | Storage for the classifier, embeddings, and metadata. |

---

## Technical Specifications

- **Dataset:** 14,174 Master+ games.
- **Model Type:** Linear Discriminant Analysis (LDA).
- **Performance:** AUC 0.5735.
- **Response Time:** Under 200ms per API request.

---

## API Reference

**Endpoint:** `POST /api/recommend`

**Request Body:**

```json
{
  "blue": {
    "picks": ["Malphite", "Vi"],
    "bans": ["Yasuo"]
  },
  "red": {
    "picks": ["Jayce"],
    "bans": ["Garen"]
  }
}
```

**Response:**

```json
{
  "recommendations": [
    { "name": "Orianna", "pct": 64 },
    { "name": "Jinx", "pct": 63 },
    { "name": "Thresh", "pct": 61 }
  ],
  "source": "ml_model"
}
```

---

## Troubleshooting

- **Static Scores (50% or 33% for each recommendations):** This shows the ML model failed to load. Ensure you have run `train_v7_fast.py` first to load the ML model.

---

## Contacts

Built by Mokhtari Younes and Boussihmed Ayman, Emerald+ EUW players passionate about data and competitive League of Legends.
Feel free to reach out for feedback, contributions, or questions.


GitHub: github.com/ayihmed-creator &  github.com/younswow  
Discord: youns1 & aylegrand  
Email: boussihmed@insa-toulouse.fr & ymokhtar@insa-toulouse.fr 
