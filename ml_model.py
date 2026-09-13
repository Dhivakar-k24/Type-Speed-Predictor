"""
ml_model.py
───────────
Two ML models for the Type Speed Predictor:

  1. RandomForestRegressor  → predicts exact WPM  (regression)
  2. KNN + RandomForest Classifier → predicts speed CLASS (classification)
     Classes: Beginner | Average | Above Average | Proficient | Fast | Expert

Both models use the same 6 features:
  char_count, word_count, space_count, duration_seconds,
  avg_word_length (derived), chars_per_second (derived)
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, r2_score,
                              accuracy_score, classification_report)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── File paths ──────────────────────────────────────────────────────────────
REG_MODEL_PATH    = "wpm_model.pkl"
REG_SCALER_PATH   = "wpm_scaler.pkl"
CLF_MODEL_PATH    = "clf_model.pkl"
CLF_SCALER_PATH   = "clf_scaler.pkl"
CLF_ENCODER_PATH  = "clf_encoder.pkl"

MIN_SESSIONS = 5

# ── Speed class definitions (ordered) ───────────────────────────────────────
SPEED_CLASSES = ["Beginner", "Average", "Above Average", "Proficient", "Fast", "Expert"]

CLASS_META = {
    "Beginner":      {"range": "< 20 WPM",   "color": "#e25c5c", "emoji": "🐢"},
    "Average":       {"range": "20–39 WPM",  "color": "#e2945c", "emoji": "🚶"},
    "Above Average": {"range": "40–59 WPM",  "color": "#e2d45c", "emoji": "🏃"},
    "Proficient":    {"range": "60–79 WPM",  "color": "#5ce27a", "emoji": "⚡"},
    "Fast":          {"range": "80–99 WPM",  "color": "#5cb8e2", "emoji": "🚀"},
    "Expert":        {"range": "100+ WPM",   "color": "#a55ce2", "emoji": "🔥"},
}


# ── Feature engineering ──────────────────────────────────────────────────────

def wpm_to_class(wpm):
    if wpm < 20:  return "Beginner"
    if wpm < 40:  return "Average"
    if wpm < 60:  return "Above Average"
    if wpm < 80:  return "Proficient"
    if wpm < 100: return "Fast"
    return "Expert"


def build_features(char_count, word_count, space_count, duration_seconds):
    avg_word_length  = char_count / max(word_count, 1)
    chars_per_second = char_count / max(duration_seconds, 1)
    return [char_count, word_count, space_count,
            duration_seconds, avg_word_length, chars_per_second]

FEATURE_NAMES = [
    "char_count", "word_count", "space_count",
    "duration_seconds", "avg_word_length", "chars_per_second"
]


# ════════════════════════════════════════════════════════════════════════════
# REGRESSION  (RandomForestRegressor → predicts WPM number)
# ════════════════════════════════════════════════════════════════════════════

def train_model(sessions):
    if len(sessions) < MIN_SESSIONS:
        return {"success": False,
                "error": f"Need at least {MIN_SESSIONS} sessions. You have {len(sessions)}."}

    X, y = [], []
    for s in sessions:
        try:
            X.append(build_features(s["char_count"], s["word_count"],
                                     s["space_count"], s["duration_seconds"]))
            y.append(s["wpm"])
        except Exception:
            continue

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)
        has_test = True
    else:
        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
        has_test = False

    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 2)
    r2  = round(float(r2_score(y_test, y_pred)), 3)

    importances = {n: round(float(v), 4)
                   for n, v in zip(FEATURE_NAMES, model.feature_importances_)}

    with open(REG_MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(REG_SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    return {"success": True, "sessions_used": len(X),
            "mae": mae, "r2": r2, "has_test_split": has_test,
            "feature_importances": importances,
            "message": f"Regressor trained on {len(X)} sessions. MAE={mae} WPM, R²={r2}"}


def predict_wpm(char_count, word_count, space_count, duration_seconds):
    if not os.path.exists(REG_MODEL_PATH):
        return {"success": False,
                "error": "Regressor not trained yet. Train first."}

    with open(REG_MODEL_PATH, "rb") as f: model = pickle.load(f)
    with open(REG_SCALER_PATH, "rb") as f: scaler = pickle.load(f)

    feats = build_features(char_count, word_count, space_count, duration_seconds)
    X = scaler.transform([feats])

    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
    predicted  = int(round(float(model.predict(X)[0])))
    low  = int(round(float(np.percentile(tree_preds, 10))))
    high = int(round(float(np.percentile(tree_preds, 90))))

    return {"success": True, "predicted_wpm": predicted,
            "confidence_range": f"{low}–{high} WPM",
            "label": wpm_to_class(predicted), "model": "RandomForestRegressor"}


def model_exists():
    return os.path.exists(REG_MODEL_PATH) and os.path.exists(REG_SCALER_PATH)


# ════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION  (KNN  vs  RandomForestClassifier → picks the better one)
# ════════════════════════════════════════════════════════════════════════════

def train_classifier(sessions):
    """
    Train both KNN and RandomForestClassifier.
    Compare using cross-validation accuracy.
    Save the winner. Return full report.
    """
    if len(sessions) < MIN_SESSIONS:
        return {"success": False,
                "error": f"Need at least {MIN_SESSIONS} sessions. You have {len(sessions)}."}

    X, y_raw = [], []
    for s in sessions:
        try:
            X.append(build_features(s["char_count"], s["word_count"],
                                     s["space_count"], s["duration_seconds"]))
            y_raw.append(wpm_to_class(s["wpm"]))
        except Exception:
            continue

    X = np.array(X)

    # Encode labels
    encoder = LabelEncoder()
    encoder.fit(SPEED_CLASSES)          # fix the class order
    y = encoder.transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Build candidate models ──────────────────────────────────────────────
    k = max(1, min(5, len(X) - 1))     # k must be < n_samples
    candidates = {
        "KNN": KNeighborsClassifier(n_neighbors=k, metric="euclidean"),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42)
    }

    # cv folds must not exceed the smallest class size
    from collections import Counter
    class_counts = Counter(y.tolist())
    min_class    = min(class_counts.values())
    cv_folds     = max(2, min(5, min_class))   # at least 2, at most 5
    cv_scores    = {}
    for name, clf in candidates.items():
        if len(X) >= 4 and min_class >= 2:
            scores = cross_val_score(clf, X_scaled, y,
                                     cv=cv_folds, scoring="accuracy")
            cv_scores[name] = round(float(scores.mean()), 3)
        else:
            # Not enough data per class for CV — train on all, score on all
            clf.fit(X_scaled, y)
            cv_scores[name] = round(float(accuracy_score(y, clf.predict(X_scaled))), 3)

    # ── Pick winner ────────────────────────────────────────────────────────
    winner_name = max(cv_scores, key=cv_scores.get)
    winner_clf  = candidates[winner_name]

    # Train winner on full data
    winner_clf.fit(X_scaled, y)

    # ── Evaluate on full set (best we can do with small data) ──────────────
    y_pred   = winner_clf.predict(X_scaled)
    accuracy = round(float(accuracy_score(y, y_pred)), 3)

    # Class distribution
    unique, counts = np.unique(y_raw, return_counts=True)
    class_dist = {cls: int(cnt) for cls, cnt in zip(unique, counts)}

    # Feature importances (RF only)
    importances = {}
    if winner_name == "RandomForest":
        importances = {n: round(float(v), 4)
                       for n, v in zip(FEATURE_NAMES, winner_clf.feature_importances_)}

    # Persist winner + scaler + encoder
    with open(CLF_MODEL_PATH,   "wb") as f: pickle.dump(winner_clf, f)
    with open(CLF_SCALER_PATH,  "wb") as f: pickle.dump(scaler, f)
    with open(CLF_ENCODER_PATH, "wb") as f: pickle.dump(encoder, f)

    return {
        "success":        True,
        "sessions_used":  len(X),
        "winner_model":   winner_name,
        "cv_scores":      cv_scores,
        "accuracy":       accuracy,
        "class_dist":     class_dist,
        "feature_importances": importances,
        "message": (f"{winner_name} won with CV accuracy "
                    f"{cv_scores[winner_name]*100:.1f}%. "
                    f"Trained on {len(X)} sessions.")
    }


def classify_session(char_count, word_count, space_count, duration_seconds):
    """
    Classify a typing session into a speed class using the trained classifier.
    Returns: predicted class, confidence %, all class probabilities, model used.
    """
    if not os.path.exists(CLF_MODEL_PATH):
        return {"success": False,
                "error": "Classifier not trained yet. Train first from the Stats tab."}

    with open(CLF_MODEL_PATH,   "rb") as f: clf     = pickle.load(f)
    with open(CLF_SCALER_PATH,  "rb") as f: scaler  = pickle.load(f)
    with open(CLF_ENCODER_PATH, "rb") as f: encoder = pickle.load(f)

    feats  = build_features(char_count, word_count, space_count, duration_seconds)
    X      = scaler.transform([feats])

    pred_encoded = clf.predict(X)[0]
    predicted_class = encoder.inverse_transform([pred_encoded])[0]

    # Confidence probabilities
    class_probs = {}
    confidence  = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        for enc_label, prob in zip(clf.classes_, proba):
            class_name = encoder.inverse_transform([enc_label])[0]
            class_probs[class_name] = round(float(prob) * 100, 1)
        confidence = class_probs.get(predicted_class, 0.0)
    else:
        confidence = 100.0
        class_probs = {predicted_class: 100.0}

    meta = CLASS_META.get(predicted_class, {})

    return {
        "success":          True,
        "predicted_class":  predicted_class,
        "confidence":       confidence,
        "wpm_range":        meta.get("range", ""),
        "emoji":            meta.get("emoji", ""),
        "color":            meta.get("color", "#ffffff"),
        "class_probs":      class_probs,
        "model_used":       type(clf).__name__,
    }


def classifier_exists():
    return os.path.exists(CLF_MODEL_PATH)
