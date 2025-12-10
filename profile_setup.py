import json
import os
from datetime import datetime

from flask import jsonify, request


def profile_setup():
    data = request.get_json()
    required_fields = ["age", "weight", "height", "gender", "activitylevel", "goal"]
    print("Data received:", data)
    missing_fields = [field for field in required_fields if field not in data or not data[field]]
    if missing_fields:
        return jsonify({"status": "error", "message": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    age = int(data.get("age"))
    weight_kg = float(data.get("weight"))
    target_weight = float(data.get("targetWeight"))
    height_ft = float(data.get("height"))
    gender = data.get("gender").lower()  # Convert to lowercase
    activity = data.get("activitylevel").lower()  # Convert to lowercase
    goal = data.get("goal").lower()  # Convert to lowercase
    timeline_str = data.get("timeline")  # Target timeline from frontend
    muscles = data.get("muscles", [])  # ⬅ NEW: muscles list
    height_m = height_ft * 0.3048
    height_cm = height_m * 100
    bmi = weight_kg / (height_m ** 2)
    if gender == "male":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    activity_multipliers = {
        "sedentary": 1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active": 1.725,
        "extra active": 1.9
    }

    tdee = bmr * activity_multipliers.get(activity.strip(), activity_multipliers)  # Default to sedentary if not found

    if bmi <= 18.5:
        suggested_goal = "weight gain"
    elif 18.5 <= bmi <= 25:
        suggested_goal = "maintain"
    else:
        suggested_goal = "weight loss"
    goal_warning = ""

    if goal != suggested_goal:
        goal_warning = f"Your selected goal '{goal}' may not match your BMI. Suggested: '{suggested_goal}'."

    if goal == "weight loss" and target_weight >= weight_kg:
        goal_warning = f"Target weight must be LOWER than current weight for weight loss."
    elif goal in ["weight gain"] and target_weight <= weight_kg:
        goal_warning = f"Target weight must be HIGHER than current weight for {goal}."
    elif goal == "maintain" and abs(target_weight - weight_kg) > 3:
        goal_warning = "For 'Stay Fit', target weight should be close to current weight (within ±3 kg)."
    warning = ""
    timeline_info = {}
    suggested_target_weight = None
    timeline_weeks = 0  # Default to 0 if timeline is not valid or not provided

    if timeline_str:
        try:
            timeline_date = datetime.strptime(timeline_str, "%Y-%m-%d")
            today = datetime.now()
            delta_days = (timeline_date - today).days

            if delta_days < 0:
                warning = "Target timeline cannot be in the past."
            elif delta_days < 7:
                warning = "Target timeline must be at least 1 week ahead."
            elif len(muscles) > 0 and delta_days < 90:
                warning = "If selecting muscles, target timeline must be at least 90 days."
            else:
                timeline_weeks = max(delta_days // 7, 0)

                # Max 1 kg per week rule
                weeks = delta_days / 7
                max_weight_change = weeks * 1  # 1 kg per week

                if abs(target_weight - weight_kg) > max_weight_change:
                    # Calculate suggested safe target
                    if target_weight > weight_kg:
                        suggested_target_weight = weight_kg + max_weight_change
                    else:
                        suggested_target_weight = weight_kg - max_weight_change

                    warning = f"Target weight change exceeds 1 kg/week. Suggested safe target: {suggested_target_weight:.1f} kg."
                else:
                    timeline_info["days"] = delta_days

        except Exception:
            warning = "Invalid timeline date format."


    data.update({
        "bmi": round(bmi, 2),
        "bmr": round(bmr, 2),
        "tdee": round(tdee, 2),
        "suggested_goal": suggested_goal,
        "goal_warning": goal_warning,
        "warning": warning,
        "timeline_info": timeline_info,
        "timeline_weeks": timeline_weeks,
        "suggested_target_weight": round(suggested_target_weight, 2) if suggested_target_weight else None
    })

    return jsonify({
        "status": "success",
        "message": "Profile saved & health metrics calculated",
        "bmi": data["bmi"],
        "bmr": data["bmr"],
        "tdee": data["tdee"],
        "selected_goal": goal,
        "suggested_goal": suggested_goal,
        "goal_warning": goal_warning,
        "warning": warning,
        "timeline_weeks": timeline_weeks,
        "timeline_info": timeline_info,
        "suggested_target_weight": data["suggested_target_weight"]
    })

