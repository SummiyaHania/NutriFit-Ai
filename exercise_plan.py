import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random
import hashlib

# ---------------- Load Dataset ----------------
exercise_df = pd.read_csv('excercise.csv')

categorical_features = ['muscle_group', 'type', 'intensity']
numerical_features = ['sets', 'repetitions', 'duration']

encoder = OneHotEncoder()
X_cat = encoder.fit_transform(exercise_df[categorical_features])

scaler = StandardScaler()
X_num = scaler.fit_transform(exercise_df[numerical_features])

X_final = np.hstack([X_cat.toarray(), X_num])

# ---------------- Activity Level Mapping ----------------
def map_activity_level(activity_level):
    """
    Map user activity level to beginner or advanced
    """
    activity_level = activity_level.lower()

    if activity_level in ['sedentary', 'lightly active']:
        return 'beginner'
    elif activity_level in ['moderately active', 'very active', 'extra active']:
        return 'advanced'
    return 'beginner'

# ---------------- Helper Functions ----------------
def create_user_vector(user_profile):
    """
    Create a user vector for similarity calculations (optional)
    """
    goal_vector = (exercise_df['target_goal'].str.lower() == user_profile['target_goal'].lower()).astype(
        int).values.reshape(-1, 1)
    numeric_placeholder = np.mean(X_num, axis=0).reshape(1, -1)
    user_vector = np.hstack([goal_vector.T, numeric_placeholder])
    return user_vector

def adjust_exercise(row, activity, target_goal, timeline_weeks):

    sets, reps, duration = row['sets'], row['repetitions'], row['duration']

    # Activity adjustment
    activity = map_activity_level(activity)
    if activity == 'beginner':
        sets *= 0.8
        reps *= 0.8
        duration *= 0.8
    elif activity == 'advanced':
        sets *= 1.2
        reps *= 1.2
        duration *= 1.2

    # Goal adjustment
    target_goal = target_goal.lower()
    if target_goal in ['weight loss', 'lose weight', 'fat loss', 'weight lose']:
        duration *= 1.2
    elif target_goal in ['weight gain', 'gain weight', 'muscle gain']:
        reps *= 1.2

    # Timeline adjustment
    if timeline_weeks <= 4:
        sets *= 1.1
        reps *= 1.1
        duration *= 1.1
    elif 5 <= timeline_weeks <= 8:
        sets *= 1.05
        reps *= 1.05
        duration *= 1.05

    return pd.Series([round(sets), round(reps), round(duration)])

# ---------------- Generate Daily Exercise Plan ----------------
def generate_exercise_plan(user_profile, days=7):
    """
    Generate a deterministic daily workout plan.
    Ensure there are no fallback random exercises; handle the 'no exercises found' case explicitly.
    Minimum 3 exercises per day.
    """
    goal = user_profile['target_goal'].lower()
    activity = user_profile['activity_level'].lower()
    timeline = user_profile['timeline_weeks']

    # Filter exercises by goal
    filtered_ex = exercise_df[exercise_df['target_goal'].str.lower() == goal].copy()

    # Handle the case where no exercises are found for the target goal
    if filtered_ex.empty:
        print(f"⚠ WARNING: No exercises found for goal '{goal}'. Please update your goal or choose another goal.")
        # Option 1: Return an empty plan with a message
        return {"error": f"No exercises found for goal '{goal}'. Please update your goal or try a different one."}

    # Adjust exercises based on the user profile
    filtered_ex[['sets', 'repetitions', 'duration']] = filtered_ex.apply(
        lambda row: adjust_exercise(row, activity, goal, timeline), axis=1)

    # ---------------- Deterministic Shuffle ----------------
    # Use a hash of the user profile as a seed for deterministic shuffling
    user_hash = int(hashlib.md5(str(user_profile).encode()).hexdigest(), 16) % (2**32)
    filtered_ex = filtered_ex.sample(frac=1, random_state=user_hash).reset_index(drop=True)

    # Determine exercises per day
    exercises_per_day = max(3, len(filtered_ex) // days)
    daily_plan = {}

    for day in range(1, days + 1):
        start_idx = (day - 1) * exercises_per_day
        end_idx = start_idx + exercises_per_day
        day_ex = filtered_ex.iloc[start_idx:end_idx]

        # Ensure at least 3 exercises per day
        if len(day_ex) < 3:
            day_ex = filtered_ex.iloc[:3]

        # Add day's exercises to plan
        daily_plan[f'Day {day}'] = day_ex[['exercise_name', 'sets', 'repetitions', 'duration']].to_dict(
            orient='records'
        )

    return daily_plan
