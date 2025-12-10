import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.optimize import nnls
import os

# ---------------------------------------------------------
# 1. FOOD CATEGORIES
# ---------------------------------------------------------
BREAKFAST_FOODS = [
    "oats", "egg omelette", "boiled egg", "cereal", "plain pan cakes",
    "chai", "lassi", "yogurt", "green tea", "paratha", "brown bread",
    "sandwich", "detox water"
]

LUNCH_FOODS = [
    "roti", "curry (kidney bean)", "mutton curry", "kofte-curry",
    "chicken pulao", "chapali kabab", "shami kabab", "chicken curry",
    "chicken sajji", "fried fish", "mughal mussallam", "chicken biryani",
    "beef biryani", "mutton biryani", "cooked lentils", "nihari",
    "fruit chaat", "mashed potato", "haleem", "russian salad",
    "vegetable salad", "butter chicken", "chicken breast",
    "chicken wing", "chicken drumstick", "chicken fried rice",
    "egg fried rice", "milk shake", "sandwich", "chickpea salad",
    "palak paneer", "mashed potato", "cooked mixed vegetable",
    "chicken corn soup", "green tea", "coffee"
]

DINNER_FOODS = LUNCH_FOODS

# ---------------------------------------------------------
# 2. LOAD FOOD DATA
# ---------------------------------------------------------
def load_food_data(path="foods.csv"):
    df = pd.read_csv(path)
    required = ["food_name", "calories", "protein_g", "carbs_g", "fat_g"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing column in foods CSV: {r}")

    df.fillna(0, inplace=True)
    df["food_name_lower"] = df["food_name"].str.lower()
    return df

# ---------------------------------------------------------
# 3. TRAIN / LOAD CALORIE MODEL
# ---------------------------------------------------------
def train_calorie_model(csv_file="pakistan_user_profiles.csv",
                        model_path="models/calorie_model.pkl"):
    df = pd.read_csv(csv_file)

    df["gender"] = df["gender"].fillna("Male")
    df["activity_level"] = df["activity_level"].fillna("Moderate")
    df["target_goal"] = df["target_goal"].fillna("maintain")

    X = df[["age", "weight_kg", "height_cm", "gender", "activity_level", "target_goal"]]
    y = df["calories"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             ["gender", "activity_level", "target_goal"]),
            ("num", "passthrough", ["age", "weight_kg", "height_cm"])
        ]
    )

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.06,
        max_depth=4,
        objective="reg:squarederror"
    )

    pipeline = Pipeline([("pre", preprocessor), ("model", model)])
    pipeline.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, model_path)
    return pipeline

def load_calorie_model(model_path="models/calorie_model.pkl"):
    return joblib.load(model_path)

# ---------------------------------------------------------
# 4. FOOD FILTERING BASED ON ALLERGIES / HEALTH
# ---------------------------------------------------------
def filter_foods(df, profile):
    allergies = [a.lower() for a in profile.get("allergies", [])]
    health = [h.lower() for h in profile.get("health_conditions", [])]

    out = df.copy()
    if len(allergies) > 0:
        pattern = "|".join(allergies)
        out = out[~out["food_name_lower"].str.contains(pattern, na=False)]

    if "diabetes" in health:
        out = out[~out["food_name_lower"].str.contains(
            "halwa|kheer|mithai|dessert|juice|milkshake|soda|sweet|pancake|cereal",
            case=False, na=False
        )]

    if "hypertension" in health:
        out = out[~out["food_name_lower"].str.contains(
            "fried|samosa|pakora|chips|paratha|biryani|nihari|haleem",
            case=False, na=False
        )]

    return out.reset_index(drop=True)

# ---------------------------------------------------------
# 5. COSINE SIMILARITY FOR FOOD SELECTION
# ---------------------------------------------------------
def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_top_foods(df, target, top_n=8):
    scores = []
    for _, row in df.iterrows():
        fv = np.array([row["protein_g"], row["carbs_g"], row["fat_g"]])
        scores.append((cosine_similarity(fv, target), row))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scores[:top_n]]

# ---------------------------------------------------------
# 6. NNLS PORTIONS
# ---------------------------------------------------------
def solve_portions(selected_foods, meal_cals, meal_pro):
    # Correct:
    A = np.array([
        [f["calories"] / 100 for f in selected_foods],
        [f["protein_g"] / 100 for f in selected_foods]
    ])  # Shape: (2, num_foods)

    b = np.array([meal_cals, meal_pro])  # Shape: (2,)

    grams, _ = nnls(A, b)  # nnls(A, b) expects (m,n) for A and (m,) for b

    grams = np.clip(grams, 60, 350)
    return grams

# ---------------------------------------------------------
# 7. MACRO AND MEAL SPLITS
# ---------------------------------------------------------
MACRO_SPLITS = {
    "weight loss": (0.40, 0.35, 0.25),
    "weight gain": (0.50, 0.30, 0.20),
    "maintain": (0.45, 0.30, 0.25)
}

MEAL_SPLITS = {
    "weight loss": {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30},
    "weight gain": {"breakfast": 0.30, "lunch": 0.45, "dinner": 0.25},
    "maintain": {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30}
}

# ---------------------------------------------------------
# 8. GENERATE WEEKLY MEAL PLAN
# ---------------------------------------------------------
def generate_meal_plan(user_profile, model, food_df, days=7, cooldown=2):
    weekly_plan = {}
    recent = {"breakfast": [], "lunch": [], "dinner": []}
    goal = user_profile["target_goal"].lower()

    for day in range(1, days + 1):
        # Predict daily calories
        X = pd.DataFrame([{
            "age": user_profile["age"],
            "weight_kg": user_profile["weight_kg"],
            "height_cm": user_profile["height_cm"],
            "gender": user_profile["gender"],
            "activity_level": user_profile["activity_level"],
            "target_goal": user_profile["target_goal"]
        }])
        daily_cals = float(model.predict(X)[0])

        # Macro targets
        cr, pr, fr = MACRO_SPLITS[goal]
        total_pro = daily_cals * pr / 4
        total_car = daily_cals * cr / 4
        total_fat = daily_cals * fr / 9

        # Filter foods
        filtered = filter_foods(food_df, user_profile)

        meals = {}
        splits = MEAL_SPLITS[goal]

        for meal, pct in splits.items():
            meal_cals = daily_cals * pct
            meal_pro = total_pro * pct
            meal_car = total_car * pct
            meal_fat = total_fat * pct
            target_vec = np.array([meal_pro, meal_car, meal_fat])

            # Meal-specific pool
            if meal == "breakfast":
                pool = filtered[filtered["food_name_lower"].isin([f.lower() for f in BREAKFAST_FOODS])]
            elif meal == "lunch":
                pool = filtered[filtered["food_name_lower"].isin([f.lower() for f in LUNCH_FOODS])]
            else:
                pool = filtered[filtered["food_name_lower"].isin([f.lower() for f in DINNER_FOODS])]

            # Avoid repeated foods
            pool = pool[~pool["food_name"].isin(recent[meal])]
            if pool.empty:
                pool = filtered

            topfoods = get_top_foods(pool, target_vec, top_n=3)
            grams = solve_portions(topfoods, meal_cals, meal_pro)

            items = []
            for g, row in zip(grams, topfoods):
                items.append({
                    "food_name": row["food_name"],
                    "grams": round(float(g), 1),
                    "calories": round(float(row["calories"] * g / 100), 1),
                    "protein_g": round(float(row["protein_g"] * g / 100), 1),
                    "carbs_g": round(float(row["carbs_g"] * g / 100), 1),
                    "fat_g": round(float(row["fat_g"] * g / 100), 1)
                })
                recent[meal].append(row["food_name"])
            recent[meal] = recent[meal][-cooldown:]
            meals[meal] = items

        weekly_plan[f"day_{day}"] = {
            "predicted_daily_calories": round(daily_cals, 1),
            "daily_macro_targets": {
                "protein_g": round(total_pro, 1),
                "carbs_g": round(total_car, 1),
                "fat_g": round(total_fat, 1)
            },
            "meals": meals
        }

    return weekly_plan
