from flask import Flask, request, jsonify, render_template
from database import init_db, save_session, get_all_sessions, get_overall_stats, delete_session
from ml_model import (train_model, predict_wpm, model_exists,
                      train_classifier, classify_session, classifier_exists)

app = Flask(__name__)
init_db()

# ─────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────
# Session Routes
# ─────────────────────────────────────────

@app.route("/api/session", methods=["POST"])
def create_session():
    data = request.get_json()
    required = ["wpm", "word_count", "space_count", "char_count", "duration_seconds"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    wpm              = int(data["wpm"])
    word_count       = int(data["word_count"])
    space_count      = int(data["space_count"])
    char_count       = int(data["char_count"])
    duration_seconds = float(data["duration_seconds"])
    top_words        = str(data.get("top_words", ""))

    if wpm < 0 or word_count < 1:
        return jsonify({"error": "Invalid session data"}), 400

    session_id = save_session(wpm, word_count, space_count,
                              char_count, duration_seconds, top_words)
    return jsonify({"success": True, "session_id": session_id,
                    "message": "Session saved successfully"}), 201


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    limit    = request.args.get("limit", 20, type=int)
    sessions = get_all_sessions(limit=limit)
    return jsonify({"sessions": sessions, "count": len(sessions)})


@app.route("/api/stats", methods=["GET"])
def overall_stats():
    return jsonify(get_overall_stats())


@app.route("/api/session/<int:session_id>", methods=["DELETE"])
def remove_session(session_id):
    if delete_session(session_id):
        return jsonify({"success": True})
    return jsonify({"error": "Session not found"}), 404


# ─────────────────────────────────────────
# Regression ML Routes
# ─────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def train():
    sessions = get_all_sessions(limit=1000)
    result   = train_model(sessions)
    return jsonify(result), (200 if result["success"] else 400)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    required = ["char_count", "word_count", "space_count", "duration_seconds"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
    result = predict_wpm(float(data["char_count"]), float(data["word_count"]),
                         float(data["space_count"]), float(data["duration_seconds"]))
    return jsonify(result), (200 if result["success"] else 400)


@app.route("/api/model/status", methods=["GET"])
def model_status():
    return jsonify({"model_trained": model_exists(),
                    "classifier_trained": classifier_exists()})


# ─────────────────────────────────────────
# Classification ML Routes
# ─────────────────────────────────────────

@app.route("/api/train/classifier", methods=["POST"])
def train_clf():
    """Train KNN + RF classifier; auto-selects the better model."""
    sessions = get_all_sessions(limit=1000)
    result   = train_classifier(sessions)
    return jsonify(result), (200 if result["success"] else 400)


@app.route("/api/classify", methods=["POST"])
def classify():
    """Classify current typing session into a speed class."""
    data = request.get_json()
    required = ["char_count", "word_count", "space_count", "duration_seconds"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
    result = classify_session(float(data["char_count"]), float(data["word_count"]),
                               float(data["space_count"]), float(data["duration_seconds"]))
    return jsonify(result), (200 if result["success"] else 400)


# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Type Speed Predictor running at http://127.0.0.1:5000")
    app.run(debug=True)
