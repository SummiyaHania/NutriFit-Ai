import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from exercise_plan import map_activity_level, generate_exercise_plan

from profile_setup import profile_setup

from meal_plan import (
    load_food_data,
    load_calorie_model,
    train_calorie_model,
    generate_meal_plan
)

app = Flask(__name__)
CORS(app)

food_data = load_food_data("foods.csv")

if os.path.exists("models/calorie_model.pkl"):
    calorie_model = load_calorie_model()
else:
    calorie_model = train_calorie_model()

@app.route("/profile_setup", methods=["POST"])
def handle_profile_setup_endpoint():
    try:
        return profile_setup()
    except Exception as e:
        app.logger.error(f"Error in profile setup: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/meal_plan", methods=["POST"])
def meal_plan_endpoint():
    try:
        user_data = request.json

        result = generate_meal_plan(
            user_profile=user_data,
            model=calorie_model,
            food_df=food_data
        )

        return jsonify(result)


    except Exception as e:

        app.logger.error("Meal plan ERROR:", exc_info=True)

        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/exercise_video", methods=["POST"])
def handle_exercise_video_endpoint():
    return jsonify({"message": "Exercise video endpoint (not implemented)"}), 200

@app.route("/exercise_plan", methods=["POST"])
def exercise_plan_endpoint():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"status": "error", "message": "No user profile sent"}), 400

        user_profile = {
            'target_goal': data.get("goal").lower(),
            'activity_level': data.get("activitylevel").lower(),
            'timeline_weeks': int(data.get("timeline_weeks"))
        }

        mapped_activity_level = map_activity_level(user_profile['activity_level'])
        user_profile['activity_level'] = mapped_activity_level

        plan = generate_exercise_plan(user_profile)

        if not plan:
            return jsonify({"status": "error", "message": "Failed to generate exercise plan"}), 500

        return jsonify({
            "status": "success",
            "message": "Exercise plan generated successfully",
            "plan": plan
        })

    except Exception as e:
        app.logger.error(f"Error generating exercise plan: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# Run App
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
