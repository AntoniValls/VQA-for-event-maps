"""
Weighted risk score (paper §III-C) and risk categories (paper §IV-C).
Single definition used by both the evaluation (MAE_R) and the event maps.
"""
import string
from typing import Dict

# Hazard tiers: weight w_i of each Level-1 question
HAZARD_CONFIG = {
    "CRITICAL": {"weight": 1.0, "ids": ["q_construction_visible", "q_surface_hazardous", "q_pedestrian_not_on_sidewalk"]},
    "HIGH":     {"weight": 0.6, "ids": ["q_crossing_nearby", "q_stairs_visible", "q_obstacle_blocking"]},
    "LOW":      {"weight": 0.3, "ids": ["q_pedestrians_present", "q_vehicle_nearby"]},
}
WEIGHTS = {qid: cfg["weight"] for cfg in HAZARD_CONFIG.values() for qid in cfg["ids"]}

# A 'No' answer subtracts w_i / |Q| (safety reward), with |Q| = 8 Level-1 questions
SAFETY_REWARD_RATIO = 1 / len(WEIGHTS)

# Upper bounds of the risk categories: Safe <= 0.15 < Caution < 0.4 <= Danger < 0.7 <= High Risk
RISK_CATEGORIES = [
    ("Safe",      "#2ecc71"),
    ("Caution",   "#f1c40f"),
    ("Danger",    "#e67e22"),
    ("High Risk", "#e74c3c"),
]


def image_risk(answers_by_question: Dict[str, Dict]) -> float:
    """
    Normalized risk R_img in [0, 1] for one image.

    Args:
        answers_by_question: {question_id: {"answer": ...}} for that image (GT or model).
                             Only the Level-1 questions are used; unanswered ones are ignored.
    """
    answered = [qid for qid in WEIGHTS if qid in answers_by_question]
    max_theoretical = sum(WEIGHTS[qid] for qid in answered)
    if max_theoretical <= 0:
        return 0.0

    net_score = 0.0
    for qid in answered:
        ans = str(answers_by_question[qid].get('answer', '')).lower().strip()
        ans = ans.translate(str.maketrans('', '', string.punctuation))

        if any(pos in ans for pos in ['yes', 'true', 'hazard']):
            net_score += WEIGHTS[qid]
        elif any(neg in ans for neg in ['no', 'false', 'safe']):
            net_score -= WEIGHTS[qid] * SAFETY_REWARD_RATIO

    return min(max(0.0, net_score) / max_theoretical, 1.0)


def risk_category(score: float) -> int:
    """Index into RISK_CATEGORIES."""
    if score <= 0.15:
        return 0
    if score < 0.4:
        return 1
    if score < 0.7:
        return 2
    return 3


def risk_color(score: float) -> str:
    return RISK_CATEGORIES[risk_category(score)][1]
