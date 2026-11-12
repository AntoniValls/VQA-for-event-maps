"""
Weighted risk score (paper §III-C) and risk categories (paper §IV-C).
Single definition used by both the evaluation (MAE_R) and the event maps.

Two scoring modes, both reachable through image_risk():
  - hierarchical=True, aggregation="noisy_or" (default): Level-2/Level-3 answers modulate
    their parent Level-1 question's severity instead of being ignored.
  - hierarchical=False, aggregation="sum": the original Level-1-only formula.
"""
import json
import math
import string
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "vqa" / "prompts.json"

# Hazard tiers: weight w_i of each Level-1 question. CRITICAL is 0.85, not 1.0 - even a fully
# escalated critical hazard (s_i=1) should leave some irreducible uncertainty in the noisy-or
# aggregation below, rather than letting a single question saturate R_img on its own.
HAZARD_CONFIG = {
    "CRITICAL": {"weight": 0.85, "ids": ["q_construction_visible", "q_surface_hazardous", "q_pedestrian_not_on_sidewalk"]},
    "HIGH":     {"weight": 0.6, "ids": ["q_crossing_nearby", "q_stairs_visible", "q_obstacle_blocking"]},
    "LOW":      {"weight": 0.3, "ids": ["q_pedestrians_present", "q_vehicle_nearby"]},
}
WEIGHTS = {qid: cfg["weight"] for cfg in HAZARD_CONFIG.values() for qid in cfg["ids"]}

# A 'No' answer subtracts w_i / |Q| (safety reward), with |Q| = 8 Level-1 questions. Used only
# by the legacy hierarchical=False, aggregation="sum" formula.
SAFETY_REWARD_RATIO = 1 / len(WEIGHTS)

# Level-1 questions have no "dependency" entry of their own (they're the hierarchy's roots), so
# their own base severity isn't in DETAIL_CONFIG below - it's fixed here instead.
BASE_SEVERITY_L1 = 0.5

# (sign, sigma) of every Level-2/Level-3 question relative to its parent (the "dependency" field
# in vqa/prompts.json): sign=+1 means answering "Yes" raises the parent's strength (escalates),
# sign=-1 means it lowers it (mitigates). sigma in [0, 1] is the question's own severity/magnitude.
DETAIL_CONFIG = {
    "q_pedestrian_on_road": (+1, 0.8),
    "q_pedestrian_on_unpaved": (+1, 0.4),

    "q_crossing_signal_present": (-1, 0.3),
    "q_crossing_signal_red_green": (+1, 0.5),
    "q_crossing_vehicles_present": (+1, 0.4),
    "q_crossing_vehicles_moving": (+1, 0.6),
    "q_crossing_marked": (-1, 0.2),
    "q_crossing_others_waiting": (-1, 0.15),

    "q_stairs_heading": (+1, 0.4),
    "q_stairs_only_option": (+1, 0.5),
    "q_stairs_handrail": (-1, 0.3),

    "q_construction_barriers": (+1, 0.6),
    "q_construction_equipment": (+1, 0.4),
    "q_construction_overhead": (+1, 0.5),

    "q_obstacle_upcoming_step": (+1, 0.4),
    "q_obstacle_is_vehicle": (+1, 0.3),
    "q_obstacle_movable_objects": (+1, 0.2),
    "q_obstacle_fixed_infrastructure": (+1, 0.2),
    "q_obstacle_blocks_half": (+1, 0.5),
    "q_obstacle_space_around": (-1, 0.4),
    "q_obstacle_overhead": (+1, 0.5),
    "q_obstacle_seating": (+1, 0.2),

    "q_crowding_same_direction": (-1, 0.3),
    "q_crowding_slow_movement": (+1, 0.5),

    "q_vehicle_is_bicycle": (+1, 0.3),
    "q_bicycle_moving": (+1, 0.4),
    "q_bicycle_bike_lane": (-1, 0.4),
    "q_vehicle_is_car_truck_bus": (+1, 0.6),
    "q_motor_vehicle_moving": (+1, 0.6),
    "q_motor_vehicle_blocking": (+1, 0.6),
    "q_vehicle_is_motorcycle": (+1, 0.5),
    "q_motorcycle_sidewalk": (+1, 0.6),
    "q_motorcycle_moving": (+1, 0.5),

    "q_surface_water_ice": (+1, 0.6),
    "q_surface_sloped": (+1, 0.4),
    "q_surface_cracks_holes": (+1, 0.6),
}

# Upper bounds of the risk categories: Safe <= 0.15 < Caution < 0.4 <= Danger < 0.7 <= High Risk
RISK_CATEGORIES = [
    ("Safe",      "#2ecc71"),
    ("Caution",   "#f1c40f"),
    ("Danger",    "#e67e22"),
    ("High Risk", "#e74c3c"),
]


def _load_children_by_parent() -> Dict[str, list]:
    """id -> [child_id, ...], built from the "dependency" field of every question in prompts.json
    (not hardcoded: adding/removing follow-up questions there is enough to update the tree)."""
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    children = {}
    for category in data["prompt_categories"].values():
        for q in category["questions"]:
            parent = q.get("dependency")
            if parent:
                children.setdefault(parent, []).append(q["id"])
    return children


CHILDREN_BY_PARENT = _load_children_by_parent()


def _clean_answer(ans_entry) -> str:
    ans = str((ans_entry or {}).get("answer", "")).lower().strip()
    return ans.translate(str.maketrans("", "", string.punctuation))


def _is_yes(ans_entry) -> bool:
    if not ans_entry:
        return False
    ans = _clean_answer(ans_entry)
    return any(pos in ans for pos in ["yes", "true", "hazard"])


def _sigma(qid: str) -> float:
    return BASE_SEVERITY_L1 if qid in WEIGHTS else DETAIL_CONFIG[qid][1]


def _node_strength(qid: str, answers_by_question: Dict[str, Dict], memo: Dict[str, float]) -> float:
    """
    s_n for node n, given n itself is answered "Yes" (the caller is responsible for checking
    that and treating the contribution as 0 otherwise - "No" and unasked children add nothing):

        s_n = [ sigma_n + (1 - sigma_n)*A_n ] * (1 - M_n)
        A_n = 1 - Pi_{c child, sign +} (1 - y_c*s_c)
        M_n = 1 - Pi_{c child, sign -} (1 - y_c*s_c)
    """
    if qid in memo:
        return memo[qid]

    pos_terms, neg_terms = [], []
    for child_id in CHILDREN_BY_PARENT.get(qid, []):
        sign, _ = DETAIL_CONFIG[child_id]
        term = _node_strength(child_id, answers_by_question, memo) if _is_yes(answers_by_question.get(child_id)) else 0.0
        (pos_terms if sign > 0 else neg_terms).append(term)

    A_n = 1 - math.prod(1 - t for t in pos_terms)
    M_n = 1 - math.prod(1 - t for t in neg_terms)
    sigma_n = _sigma(qid)
    s_n = (sigma_n + (1 - sigma_n) * A_n) * (1 - M_n)

    memo[qid] = s_n
    return s_n


def _image_risk_noisy_or(answers_by_question: Dict[str, Dict]) -> float:
    """R_img = 1 - Pi_i (1 - w_i*y_i*s_i) over the 8 Level-1 questions."""
    memo: Dict[str, float] = {}
    terms = []
    for qid, w in WEIGHTS.items():
        s_i = _node_strength(qid, answers_by_question, memo) if _is_yes(answers_by_question.get(qid)) else 0.0
        terms.append(w * s_i)
    return 1 - math.prod(1 - t for t in terms)


def _image_risk_sum(answers_by_question: Dict[str, Dict]) -> float:
    """Original formula: only the 8 Level-1 questions, "No" rewards -w_i/|Q|, clamped to [0, 1]."""
    answered = [qid for qid in WEIGHTS if qid in answers_by_question]
    max_theoretical = sum(WEIGHTS[qid] for qid in answered)
    if max_theoretical <= 0:
        return 0.0

    net_score = 0.0
    for qid in answered:
        ans_entry = answers_by_question[qid]
        ans = _clean_answer(ans_entry)
        if any(pos in ans for pos in ["yes", "true", "hazard"]):
            net_score += WEIGHTS[qid]
        elif any(neg in ans for neg in ["no", "false", "safe"]):
            net_score -= WEIGHTS[qid] * SAFETY_REWARD_RATIO

    return min(max(0.0, net_score) / max_theoretical, 1.0)


def image_risk(answers_by_question: Dict[str, Dict], hierarchical: bool = True, aggregation: str = "noisy_or") -> float:
    """
    Normalized risk R_img in [0, 1] for one image.

    Args:
        answers_by_question: {question_id: {"answer": ...}} for that image (GT or model).
        hierarchical: if True (default), Level-2/Level-3 answers modulate their parent
            Level-1 question's severity (see _node_strength). If False, only the 8 Level-1
            questions count, exactly as before this was added.
        aggregation: "noisy_or" (the only mode implemented for hierarchical=True) or "sum"
            (the only mode implemented for hierarchical=False, i.e. the original formula).
    """
    if not hierarchical:
        if aggregation != "sum":
            raise ValueError("hierarchical=False only supports aggregation='sum'")
        return _image_risk_sum(answers_by_question)

    if aggregation != "noisy_or":
        raise ValueError("hierarchical=True only supports aggregation='noisy_or'")
    return _image_risk_noisy_or(answers_by_question)


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
