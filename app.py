import os
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_PROBLEMS_PATH = os.path.join(BASE_DIR, "all_problems_data.json")
MODEL_DATA_PATH = os.path.join(BASE_DIR, "model_data.json")
TSMC_DATA_PATH = os.path.join(BASE_DIR, "tsmc_model_data.json")

PROBLEMS_REGISTRY = {}

if os.path.exists(ALL_PROBLEMS_PATH):
    try:
        with open(ALL_PROBLEMS_PATH, "r", encoding="utf-8") as f:
            PROBLEMS_REGISTRY = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load all_problems_data.json: {e}")

if not PROBLEMS_REGISTRY:
    startups_data = {}
    if os.path.exists(MODEL_DATA_PATH):
        with open(MODEL_DATA_PATH, "r", encoding="utf-8") as f:
            startups_data = json.load(f)
    tsmc_data = {}
    if os.path.exists(TSMC_DATA_PATH):
        with open(TSMC_DATA_PATH, "r", encoding="utf-8") as f:
            tsmc_data = json.load(f)
            
    PROBLEMS_REGISTRY = {
        "50_startups": {"id": "50_startups", "name": "50 Startups Profit Prediction", "data": startups_data},
        "tsmc_2330": {"id": "tsmc_2330", "name": "TSMC (2330.TW) Stock Auto-Regression", "data": tsmc_data}
    }

MODEL_DATA = PROBLEMS_REGISTRY.get("50_startups", {}).get("data", {})


@app.route("/")
def index():
    """Render the main regression explorer dashboard with problem dropdown & dataset analyzer."""
    return render_template("index.html")


@app.route("/api/problems", methods=["GET"])
def get_problems():
    """Return available regression problem definitions."""
    summary = {}
    for k, v in PROBLEMS_REGISTRY.items():
        summary[k] = {
            "id": v.get("id"),
            "name": v.get("name"),
            "category": v.get("category"),
            "targetName": v.get("targetName"),
            "sweetSpotK": v.get("sweetSpotK")
        }
    return jsonify(summary)


@app.route("/api/models", methods=["GET"])
def get_models():
    """Return feature selection evaluation data for a given problem and algorithm."""
    problem = request.args.get("problem", "50_startups")
    prob_entry = PROBLEMS_REGISTRY.get(problem, PROBLEMS_REGISTRY.get("50_startups", {}))
    data = prob_entry.get("data", {})
    
    algo = request.args.get("algo")
    if algo and algo in data:
        return jsonify({algo: data[algo]})
    return jsonify(data)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Live prediction endpoint: Calculates output based on user inputs
    for either 50 Startups or TSMC 2330.TW Auto-Regression.
    """
    try:
        req = request.get_json(force=True) or {}
        problem = req.get("problem", "50_startups")
        algo = req.get("algo", "Lasso")
        k = int(req.get("k", 2))

        prob_entry = PROBLEMS_REGISTRY.get(problem, PROBLEMS_REGISTRY.get("50_startups", {}))
        data = prob_entry.get("data", {})

        if algo not in data or k < 1 or k > len(data[algo]):
            return jsonify({"error": "Invalid algorithm or k parameter"}), 400

        model_spec = data[algo][k - 1]
        coefficients = model_spec["coefficients"]

        predicted_val = 0.0
        breakdown = []

        if problem == "tsmc_2330":
            lag1 = float(req.get("lag1", 820))
            ma5 = float(req.get("ma5", 815))
            lag2 = float(req.get("lag2", 810))
            lag3 = float(req.get("lag3", 805))
            lag5 = float(req.get("lag5", 800))
            val_map = {"Lag_1": lag1, "MA_5": ma5, "Lag_2": lag2, "Lag_3": lag3, "Lag_5": lag5}

            for c in coefficients:
                feat = c["feature"]
                coef = c["coef"]
                if "Intercept" in feat:
                    predicted_val += coef
                    breakdown.append({"feature": "Intercept", "coef": coef, "val": 1.0, "contrib": coef})
                elif feat in val_map:
                    contrib = coef * val_map[feat]
                    predicted_val += contrib
                    breakdown.append({"feature": feat, "coef": coef, "val": val_map[feat], "contrib": contrib})
        else:
            rd_spend = float(req.get("rd_spend", 100000))
            marketing_spend = float(req.get("marketing_spend", 150000))
            admin_spend = float(req.get("admin_spend", 120000))
            state = req.get("state", "California")

            for c in coefficients:
                feat = c["feature"]
                coef = c["coef"]
                if "Intercept" in feat:
                    predicted_val += coef
                    breakdown.append({"feature": "Intercept", "coef": coef, "val": 1.0, "contrib": coef})
                elif feat == "R&D Spend":
                    contrib = coef * rd_spend
                    predicted_val += contrib
                    breakdown.append({"feature": "R&D Spend", "coef": coef, "val": rd_spend, "contrib": contrib})
                elif feat == "Marketing Spend":
                    contrib = coef * marketing_spend
                    predicted_val += contrib
                    breakdown.append({"feature": "Marketing Spend", "coef": coef, "val": marketing_spend, "contrib": contrib})
                elif feat == "Administration":
                    contrib = coef * admin_spend
                    predicted_val += contrib
                    breakdown.append({"feature": "Administration", "coef": coef, "val": admin_spend, "contrib": contrib})
                elif feat == "State_Florida":
                    val = 1.0 if state == "Florida" else 0.0
                    contrib = coef * val
                    predicted_val += contrib
                    breakdown.append({"feature": "State_Florida", "coef": coef, "val": val, "contrib": contrib})
                elif feat == "State_New York":
                    val = 1.0 if state == "New York" else 0.0
                    contrib = coef * val
                    predicted_val += contrib
                    breakdown.append({"feature": "State_New York", "coef": coef, "val": val, "contrib": contrib})

        return jsonify({
            "predicted_value": round(predicted_val, 2),
            "breakdown": breakdown,
            "features_used": model_spec["features"],
            "k": k,
            "algorithm": algo,
            "problem": problem
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
