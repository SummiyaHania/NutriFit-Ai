import math
import warnings

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from pydantic import BaseModel

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.figsize"] = (14, 5)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Enums  (identical to original)
# ══════════════════════════════════════════════════════════════════════════════

class Goal(Enum):
    WEIGHT_LOSS  = "weight_loss"
    MUSCLE_GAIN  = "muscle_gain"
    MAINTENANCE  = "maintenance"


class ActivityLevel(Enum):
    SEDENTARY   = 1.2
    LIGHT       = 1.375
    MODERATE    = 1.55
    ACTIVE      = 1.725
    VERY_ACTIVE = 1.9


class StrategyType(Enum):
    CALORIE_SPREAD    = "calorie_spread"
    WORKOUT_BURN      = "workout_burn"
    MACRO_REBALANCE   = "macro_rebalance"
    INTERMITTENT_FAST = "intermittent_fast"
    HYBRID            = "hybrid"


class MealType(Enum):
    BREAKFAST = "breakfast"
    LUNCH     = "lunch"
    DINNER    = "dinner"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Safety constants & goal config  (identical to original)
# ══════════════════════════════════════════════════════════════════════════════

MIN_DAILY_CALORIES  = 1200
MAX_REDUCTION_PCT   = 0.20
BURN_RATE_KCAL_MIN  = 7.0
MAX_WORKOUT_MIN_DAY = 90
CHEAT_THRESHOLD_PCT = 0.15

GOAL_CONFIG = {
    Goal.WEIGHT_LOSS: {
        "calorie_multiplier": 0.80,
        "protein_pct"       : 0.35,
        "carbs_pct"         : 0.35,
        "fat_pct"           : 0.30,
        "max_window_days"   : 7,
        "urgency_factor"    : 1.3,
    },
    Goal.MUSCLE_GAIN: {
        "calorie_multiplier": 1.10,
        "protein_pct"       : 0.40,
        "carbs_pct"         : 0.40,
        "fat_pct"           : 0.20,
        "max_window_days"   : 5,
        "urgency_factor"    : 0.8,
    },
    Goal.MAINTENANCE: {
        "calorie_multiplier": 1.00,
        "protein_pct"       : 0.30,
        "carbs_pct"         : 0.40,
        "fat_pct"           : 0.30,
        "max_window_days"   : 5,
        "urgency_factor"    : 1.0,
    },
}

MEAL_DISTRIBUTION = {
    Goal.WEIGHT_LOSS : {"breakfast": 0.25, "lunch": 0.40, "dinner": 0.35},
    Goal.MUSCLE_GAIN : {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30},
    Goal.MAINTENANCE : {"breakfast": 0.30, "lunch": 0.35, "dinner": 0.35},
}

GOAL_TIP = {
    Goal.WEIGHT_LOSS : "Focus on high-protein, low-carb meals. Avoid processed sugar.",
    Goal.MUSCLE_GAIN : "Keep protein high. Do not skip meals — just add cardio.",
    Goal.MAINTENANCE : "Stay close to your macros. One extra workout handles the rest.",
}

print("✅ Enums and config ready.")
print(f"   Goals      : {[g.value for g in Goal]}")
print(f"   Strategies : {[s.value for s in StrategyType]}")
print(f"   Safety floor    : {MIN_DAILY_CALORIES} kcal/day")
print(f"   Max daily cut   : {int(MAX_REDUCTION_PCT*100)}%")
print(f"   Cheat threshold : {int(CHEAT_THRESHOLD_PCT*100)}% over target")
print(f"   Max workout/day : {MAX_WORKOUT_MIN_DAY} min")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Dataclasses  (identical to original — none dropped)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    """
    Stores user biometrics and fitness goal.
    Auto-calculates BMR → TDEE → daily calorie target and macro splits
    using the Mifflin-St Jeor equation on initialisation.
    Falls back to Firestore-provided values when available.
    """
    user_id        : str
    name           : str
    age            : int
    weight_kg      : float
    height_cm      : float
    gender         : str
    goal           : Goal
    activity_level : ActivityLevel
    cheat_frequency: int   = 1

    bmi            : float = 0.0
    bmr            : float = field(default=0.0)
    tdee           : float = field(default=0.0)
    daily_calories : float = field(default=0.0)
    daily_protein_g: float = field(default=0.0)
    daily_carbs_g  : float = field(default=0.0)
    daily_fat_g    : float = field(default=0.0)

    def __post_init__(self):
        if self.bmr == 0.0:
            if self.gender.lower() == "male":
                self.bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + 5
            else:
                self.bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age - 161

        if self.tdee == 0.0:
            self.tdee = self.bmr * self.activity_level.value

        cfg = GOAL_CONFIG[self.goal]
        if self.daily_calories == 0.0:
            self.daily_calories   = round(self.tdee * cfg["calorie_multiplier"], 1)
            self.daily_protein_g  = round((self.daily_calories * cfg["protein_pct"]) / 4, 1)
            self.daily_carbs_g    = round((self.daily_calories * cfg["carbs_pct"])   / 4, 1)
            self.daily_fat_g      = round((self.daily_calories * cfg["fat_pct"])     / 9, 1)
        elif self.daily_protein_g == 0.0:
            self.daily_protein_g = round((self.daily_calories * cfg["protein_pct"]) / 4, 1)
            self.daily_carbs_g   = round((self.daily_calories * cfg["carbs_pct"])   / 4, 1)
            self.daily_fat_g     = round((self.daily_calories * cfg["fat_pct"])     / 9, 1)

    def display(self):
        print(f'\n{"="*52}')
        print(f"  👤 {self.name} ({self.user_id})")
        print(f'{"="*52}')
        print(f"  Age / Gender    : {self.age} / {self.gender}")
        print(f"  Weight / Height : {self.weight_kg}kg / {self.height_cm}cm")
        print(f"  Goal            : {self.goal.value}")
        print(f"  Activity Level  : {self.activity_level.name}")
        print(f"  Cheat Frequency : {self.cheat_frequency}x / week")
        print(f"  {'─'*47}")
        print(f"  BMR             : {self.bmr:.0f} kcal")
        print(f"  TDEE            : {self.tdee:.0f} kcal")
        print(f"  Daily Target    : {self.daily_calories:.0f} kcal")
        print(f"  Macros P/C/F    : {self.daily_protein_g}g / {self.daily_carbs_g}g / {self.daily_fat_g}g")
        print(f'{"="*52}')


@dataclass
class MealLog:
    """A single meal entry logged by the user."""
    meal_type : MealType
    calories  : float
    protein_g : float
    carbs_g   : float
    fat_g     : float
    food_desc : str = ""


@dataclass
class DayLog:
    """All meals consumed on a given date, with computed daily totals."""
    date  : date
    meals : List[MealLog] = field(default_factory=list)

    @property
    def total_calories(self) -> float:
        return sum(m.calories for m in self.meals)

    @property
    def total_protein(self) -> float:
        return sum(m.protein_g for m in self.meals)

    @property
    def total_carbs(self) -> float:
        return sum(m.carbs_g for m in self.meals)

    @property
    def total_fat(self) -> float:
        return sum(m.fat_g for m in self.meals)

    def display(self):
        print(f"\n  📋 Day Log — {self.date}")
        print(f'  {"─"*40}')
        for m in self.meals:
            print(f"  {m.meal_type.value.capitalize():<12}: {m.calories:.0f} kcal  "
                  f"| P:{m.protein_g}g  C:{m.carbs_g}g  F:{m.fat_g}g"
                  + (f"  [{m.food_desc}]" if m.food_desc else ""))
        print(f'  {"─"*40}')
        print(f"  TOTAL        : {self.total_calories:.0f} kcal  "
              f"| P:{self.total_protein:.0f}g  C:{self.total_carbs:.0f}g  F:{self.total_fat:.0f}g")


@dataclass
class MealPlan:
    """A single planned meal in the compensation window."""
    meal_type         : MealType
    adjusted_calories : float
    protein_g         : float
    carbs_g           : float
    fat_g             : float
    note              : str = ""


@dataclass
class DayPlan:
    """One compensation day: 3 planned meals + optional exercise."""
    date             : date
    meals            : List[MealPlan] = field(default_factory=list)
    extra_workout_min: int  = 0
    day_note         : str  = ""

    @property
    def total_calories(self) -> float:
        return sum(m.adjusted_calories for m in self.meals)


@dataclass
class CompensationPlan:
    """The complete multi-day compensation plan returned to the user."""
    user_id         : str
    user_name       : str
    goal            : Goal
    cheat_date      : date
    calorie_surplus : float
    severity        : str
    strategy_used   : StrategyType
    window_days     : int
    days            : List[DayPlan] = field(default_factory=list)
    warnings        : List[str]     = field(default_factory=list)
    ml_confidence   : float         = 0.0

    def summary(self):
        """Full tabular plan — identical to original notebook output."""
        sev_icon = {"mild": "🟡", "moderate": "🟠", "severe": "🔴"}
        print(f'\n{"="*68}')
        print("  🏋️  PERSONALIZED COMPENSATION PLAN")
        print(f'{"="*68}')
        print(f"  User        : {self.user_name} ({self.user_id})")
        print(f"  Goal        : {self.goal.value}")
        print(f"  Cheat Date  : {self.cheat_date}")
        print(f"  Surplus     : {self.calorie_surplus:.0f} kcal  "
              f"{sev_icon.get(self.severity,'')} {self.severity.upper()}")
        print(f"  Strategy    : {self.strategy_used.value}  "
              f"(ML confidence: {self.ml_confidence*100:.1f}%)")
        print(f"  Window      : {self.window_days} day(s)  <- dynamically calculated")
        if self.warnings:
            print("\n  ⚠️  Warnings:")
            for w in self.warnings:
                print(f"     * {w}")
        print(f'\n  {"─"*68}')
        print(f'  {"Date":<13} {"Meal":<12} {"Cal":>6} {"Prot":>7} {"Carbs":>7} {"Fat":>6}')
        print(f'  {"─"*68}')
        for day in self.days:
            for i, meal in enumerate(day.meals):
                d_str = day.date.isoformat() if i == 0 else ""
                print(f"  {d_str:<13} {meal.meal_type.value:<12}"
                      f" {meal.adjusted_calories:>6.0f}"
                      f" {meal.protein_g:>6.1f}g"
                      f" {meal.carbs_g:>6.1f}g"
                      f" {meal.fat_g:>5.1f}g")
            print(f'  {"":13} {"-- TOTAL --":<12}'
                  f" {day.total_calories:>6.0f}  <- {day.day_note}")
            if day.extra_workout_min > 0:
                kcal_burn = day.extra_workout_min * BURN_RATE_KCAL_MIN
                print(f'  {"":13} {"🏃 EXERCISE":<12}'
                      f" +{day.extra_workout_min} min cardio  (~{kcal_burn:.0f} kcal burned)")
            print(f'  {"─"*68}')
        print()


print("✅ Models defined.")
print("   UserProfile      → auto-computes BMR, TDEE, macros")
print("   DayLog           → total_calories / protein / carbs / fat")
print("   CompensationPlan → .summary() prints full tabular plan")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Synthetic training data  (identical to original)
# ══════════════════════════════════════════════════════════════════════════════

np.random.seed(42)
N_SAMPLES = 2000

goal_values     = [g.value for g in Goal]
activity_values = [a.value for a in ActivityLevel]

records = []
for _ in range(N_SAMPLES):
    goal        = np.random.choice(goal_values)
    activity    = np.random.choice(activity_values)
    surplus_pct = np.random.uniform(0.10, 1.50)
    surplus_cal = np.random.uniform(150, 3500)
    cheat_freq  = np.random.randint(1, 6)
    age         = np.random.randint(18, 65)

    if surplus_pct > 0.80 or surplus_cal > 2000:
        label = "hybrid"
    elif goal == "muscle_gain" and activity >= 1.55:
        label = "workout_burn"
    elif goal == "weight_loss" and surplus_pct <= 0.60:
        label = "calorie_spread"
    elif goal == "maintenance" and surplus_pct <= 0.25:
        label = "macro_rebalance"
    elif surplus_pct <= 0.30 and cheat_freq <= 2:
        label = "intermittent_fast"
    elif goal == "weight_loss":
        label = "calorie_spread"
    else:
        label = "hybrid"

    records.append({
        "goal"       : goal,
        "activity"   : activity,
        "surplus_pct": surplus_pct,
        "surplus_cal": surplus_cal,
        "cheat_freq" : cheat_freq,
        "age"        : age,
        "label"      : label,
    })

df = pd.DataFrame(records)

le_goal     = LabelEncoder().fit(df["goal"])
le_activity = LabelEncoder().fit(df["activity"])
le_label    = LabelEncoder().fit(df["label"])

df["goal_enc"]     = le_goal.transform(df["goal"])
df["activity_enc"] = le_activity.transform(df["activity"])
df["label_enc"]    = le_label.transform(df["label"])

print("✅ Training data ready.")
print(f"   Samples : {N_SAMPLES}")
print("   Strategy distribution:")
print(df["label"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Train Random Forest  (identical to original, diagnostics kept)
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = ["surplus_pct", "surplus_cal", "goal_enc", "activity_enc", "cheat_freq", "age"]
TARGET   = "label_enc"

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = 12,
    min_samples_leaf = 4,
    class_weight     = "balanced",
    random_state     = 42,
    n_jobs           = -1,
)
rf.fit(X_train_s, y_train)

y_pred = rf.predict(X_test_s)
acc    = accuracy_score(y_test, y_pred)
cv     = cross_val_score(rf, X_train_s, y_train, cv=5, scoring="accuracy")

print("=" * 52)
print("  🤖 RANDOM FOREST — RESULTS")
print("=" * 52)
print(f"  Test Accuracy  : {acc*100:.2f}%")
print(f"  5-Fold CV Mean : {cv.mean()*100:.2f}%")
print(f"  5-Fold CV Std  : +-{cv.std()*100:.2f}%")
print()
print(classification_report(y_test, y_pred, target_names=le_label.classes_))

# ML metrics chart saved to disk (no plt.show on a server)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cm_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_label.classes_,
            yticklabels=le_label.classes_, ax=axes[0])
axes[0].set_title("Confusion Matrix — Strategy Prediction", fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].tick_params(axis="x", rotation=30)

importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
importances.plot(kind="barh", ax=axes[1],
                 color=sns.color_palette("muted", len(importances)))
axes[1].set_title("Feature Importances", fontweight="bold")
axes[1].set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("ml_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ ML metrics chart saved → ml_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Strategies  (identical to original — all 5 preserved)
# ══════════════════════════════════════════════════════════════════════════════

def _classify_severity(surplus: float, profile: UserProfile) -> str:
    pct = surplus / profile.daily_calories
    if pct < 0.25: return "mild"
    if pct < 0.60: return "moderate"
    return "severe"


class CompensationStrategy(ABC):
    """Abstract base — shared helpers: window calc, safety cap, meal builder."""

    def _compute_window(self, profile: UserProfile, surplus: float) -> int:
        cfg      = GOAL_CONFIG[profile.goal]
        max_cut  = profile.daily_calories * MAX_REDUCTION_PCT
        urgency  = cfg["urgency_factor"]
        max_win  = cfg["max_window_days"]
        min_days = math.ceil((surplus / max_cut) * urgency)
        return max(1, min(min_days, max_win))

    def _safe_cut(self, target_cal: float, desired_cut: float) -> float:
        cut = min(desired_cut, target_cal * MAX_REDUCTION_PCT)
        if target_cal - cut < MIN_DAILY_CALORIES:
            cut = target_cal - MIN_DAILY_CALORIES
        return max(0.0, cut)

    def _build_meals(self, profile: UserProfile, day_calories: float) -> List[MealPlan]:
        dist  = MEAL_DISTRIBUTION[profile.goal]
        cfg   = GOAL_CONFIG[profile.goal]
        meals = []
        for mt in MealType:
            m_cal = day_calories * dist[mt.value]
            meals.append(MealPlan(
                meal_type         = mt,
                adjusted_calories = round(m_cal, 1),
                protein_g         = round((m_cal * cfg["protein_pct"]) / 4, 1),
                carbs_g           = round((m_cal * cfg["carbs_pct"])   / 4, 1),
                fat_g             = round((m_cal * cfg["fat_pct"])     / 9, 1),
            ))
        return meals

    def _make_dates(self, cheat_date: date, window: int) -> List[date]:
        return [cheat_date + timedelta(days=i + 1) for i in range(window)]

    @abstractmethod
    def build_plan(self, profile, cheat_date, surplus, strategy) -> CompensationPlan:
        ...


class CalorieSpreadStrategy(CompensationStrategy):
    """Spreads the surplus deficit evenly across multiple days."""
    def build_plan(self, profile, cheat_date, surplus, strategy):
        window      = self._compute_window(profile, surplus)
        desired_cut = surplus / window
        actual_cut  = self._safe_cut(profile.daily_calories, desired_cut)
        warnings    = []
        if actual_cut < desired_cut - 1:
            warnings.append(
                f"Safety cap applied: max cut is {actual_cut:.0f} kcal/day. "
                f"Full surplus may not be compensated within {window} days."
            )
        days = []
        for d in self._make_dates(cheat_date, window):
            day_cal = profile.daily_calories - actual_cut
            days.append(DayPlan(date=d, meals=self._build_meals(profile, day_cal),
                                day_note=f"-{actual_cut:.0f} kcal deficit"))
        return CompensationPlan(
            user_id=profile.user_id, user_name=profile.name, goal=profile.goal,
            cheat_date=cheat_date, calorie_surplus=surplus,
            severity=_classify_severity(surplus, profile),
            strategy_used=strategy, window_days=window, days=days, warnings=warnings,
        )


class WorkoutBurnStrategy(CompensationStrategy):
    """Burns surplus through additional cardio. Food intake stays at target."""
    def build_plan(self, profile, cheat_date, surplus, strategy):
        window      = self._compute_window(profile, surplus)
        per_day_min = int((surplus / BURN_RATE_KCAL_MIN) / window)
        warnings    = []
        if per_day_min > MAX_WORKOUT_MIN_DAY:
            per_day_min = MAX_WORKOUT_MIN_DAY
            warnings.append(
                f"Workout capped at {MAX_WORKOUT_MIN_DAY} min/day. "
                "Extend window or switch to Hybrid for full compensation."
            )
        days = []
        for d in self._make_dates(cheat_date, window):
            days.append(DayPlan(date=d, meals=self._build_meals(profile, profile.daily_calories),
                                extra_workout_min=per_day_min,
                                day_note=f"+{per_day_min} min cardio"))
        return CompensationPlan(
            user_id=profile.user_id, user_name=profile.name, goal=profile.goal,
            cheat_date=cheat_date, calorie_surplus=surplus,
            severity=_classify_severity(surplus, profile),
            strategy_used=strategy, window_days=window, days=days, warnings=warnings,
        )


class MacroRebalanceStrategy(CompensationStrategy):
    """Shifts macros toward higher protein / lower carbs and fat."""
    def build_plan(self, profile, cheat_date, surplus, strategy):
        window   = self._compute_window(profile, surplus)
        warnings = []
        if surplus > 1500:
            warnings.append(
                "Large surplus — macro rebalancing alone has limited effect. "
                "Consider Hybrid strategy."
            )
        dist = MEAL_DISTRIBUTION[profile.goal]
        days = []
        for d in self._make_dates(cheat_date, window):
            meals = []
            for mt in MealType:
                m_cal = profile.daily_calories * dist[mt.value]
                meals.append(MealPlan(
                    meal_type         = mt,
                    adjusted_calories = round(m_cal, 1),
                    protein_g = round(profile.daily_protein_g * dist[mt.value] * 1.15, 1),
                    carbs_g   = round(profile.daily_carbs_g   * dist[mt.value] * 0.90, 1),
                    fat_g     = round(profile.daily_fat_g     * dist[mt.value] * 0.90, 1),
                    note      = "High-protein rebalance",
                ))
            days.append(DayPlan(date=d, meals=meals,
                                day_note="Macros shifted: protein up, carbs and fat down"))
        return CompensationPlan(
            user_id=profile.user_id, user_name=profile.name, goal=profile.goal,
            cheat_date=cheat_date, calorie_surplus=surplus,
            severity=_classify_severity(surplus, profile),
            strategy_used=strategy, window_days=window, days=days, warnings=warnings,
        )


class IntermittentFastStrategy(CompensationStrategy):
    """Applies 16:8 fasting — skips breakfast, splits calories between lunch & dinner."""
    def build_plan(self, profile, cheat_date, surplus, strategy):
        window = self._compute_window(profile, surplus)
        cfg    = GOAL_CONFIG[profile.goal]
        days   = []
        for d in self._make_dates(cheat_date, window):
            daily_cut = self._safe_cut(profile.daily_calories, surplus / window)
            day_cal   = max(MIN_DAILY_CALORIES, profile.daily_calories - daily_cut)
            half      = day_cal / 2
            meals = [
                MealPlan(meal_type=MealType.BREAKFAST, adjusted_calories=0,
                         protein_g=0, carbs_g=0, fat_g=0,
                         note="Skip — 16:8 fasting window"),
                MealPlan(meal_type=MealType.LUNCH, adjusted_calories=round(half, 1),
                         protein_g=round((half * cfg["protein_pct"]) / 4, 1),
                         carbs_g  =round((half * cfg["carbs_pct"])   / 4, 1),
                         fat_g    =round((half * cfg["fat_pct"])     / 9, 1),
                         note="Post-fast first meal — break fast gently"),
                MealPlan(meal_type=MealType.DINNER, adjusted_calories=round(half, 1),
                         protein_g=round((half * cfg["protein_pct"]) / 4, 1),
                         carbs_g  =round((half * cfg["carbs_pct"])   / 4, 1),
                         fat_g    =round((half * cfg["fat_pct"])     / 9, 1),
                         note="Final meal — stop eating by 8 pm"),
            ]
            days.append(DayPlan(date=d, meals=meals,
                                day_note=f"16:8 IF — {day_cal:.0f} kcal (lunch + dinner only)"))
        return CompensationPlan(
            user_id=profile.user_id, user_name=profile.name, goal=profile.goal,
            cheat_date=cheat_date, calorie_surplus=surplus,
            severity=_classify_severity(surplus, profile),
            strategy_used=strategy, window_days=window, days=days, warnings=[],
        )


class HybridStrategy(CompensationStrategy):
    """Combines calorie reduction (50%) + cardio burn (50%)."""
    def build_plan(self, profile, cheat_date, surplus, strategy):
        window      = self._compute_window(profile, surplus)
        half        = surplus / 2
        daily_cut   = self._safe_cut(profile.daily_calories, half / window)
        workout_min = min(int((half / window) / BURN_RATE_KCAL_MIN), MAX_WORKOUT_MIN_DAY)
        days = []
        for d in self._make_dates(cheat_date, window):
            day_cal = max(MIN_DAILY_CALORIES, profile.daily_calories - daily_cut)
            days.append(DayPlan(date=d, meals=self._build_meals(profile, day_cal),
                                extra_workout_min=workout_min,
                                day_note=f"-{daily_cut:.0f} kcal + {workout_min} min cardio"))
        return CompensationPlan(
            user_id=profile.user_id, user_name=profile.name, goal=profile.goal,
            cheat_date=cheat_date, calorie_surplus=surplus,
            severity=_classify_severity(surplus, profile),
            strategy_used=strategy, window_days=window, days=days, warnings=[],
        )


STRATEGY_REGISTRY = {
    StrategyType.CALORIE_SPREAD    : CalorieSpreadStrategy,
    StrategyType.WORKOUT_BURN      : WorkoutBurnStrategy,
    StrategyType.MACRO_REBALANCE   : MacroRebalanceStrategy,
    StrategyType.INTERMITTENT_FAST : IntermittentFastStrategy,
    StrategyType.HYBRID            : HybridStrategy,
}

print("✅ All 5 strategies defined and registered.")
for st, cls in STRATEGY_REGISTRY.items():
    print(f"   {st.value:<22} -> {cls.__name__}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — print_meal_compensation_plan  (identical to original — restored)
# ══════════════════════════════════════════════════════════════════════════════

def print_meal_compensation_plan(plan: CompensationPlan, profile: UserProfile):
    """
    Prints a detailed 3-meal compensation plan for every day in the window.
    Shows Breakfast/Lunch/Dinner with calories + macros, exercise row,
    cumulative recovery progress bar, and final compensation status.
    """
    sev_icon = {"mild": "🟡", "moderate": "🟠", "severe": "🔴"}
    meal_icon = {
        MealType.BREAKFAST : "🌅",
        MealType.LUNCH     : "☀️ ",
        MealType.DINNER    : "🌙",
    }
    SEP  = "=" * 62
    SEP2 = "-" * 62

    print()
    print(SEP)
    print("  🍽️  MEAL COMPENSATION PLAN")
    print(SEP)
    print(f"  User     : {plan.user_name}  ({plan.user_id})")
    print(f"  Goal     : {plan.goal.value}")
    print(f"  Surplus  : {plan.calorie_surplus:.0f} kcal  "
          f"{sev_icon.get(plan.severity, '')} {plan.severity.upper()}")
    print(f"  Strategy : {plan.strategy_used.value}  "
          f"(ML confidence: {plan.ml_confidence*100:.1f}%)")
    print(f"  Window   : {plan.window_days} compensation day(s)")
    print(f"  Tip      : {GOAL_TIP.get(plan.goal, '')}")

    if plan.warnings:
        print()
        for w in plan.warnings:
            print(f"  ⚠️  {w}")

    cumulative = 0.0

    for day_idx, day in enumerate(plan.days, start=1):
        print()
        print(SEP2)
        label = day.date.strftime("%A, %d %B %Y")
        print(f"  📅 DAY {day_idx} of {plan.window_days}  —  {label}")
        print(SEP2)

        for meal in day.meals:
            icon  = meal_icon.get(meal.meal_type, " ")
            mname = meal.meal_type.value.upper()
            print(f"  {icon} {mname}")
            if meal.adjusted_calories == 0:
                print("     Fasting — skip this meal (16:8 protocol)")
            else:
                print(f"     Calories : {meal.adjusted_calories:.0f} kcal")
                print(f"     Protein  : {meal.protein_g:.1f} g")
                print(f"     Carbs    : {meal.carbs_g:.1f} g")
                print(f"     Fat      : {meal.fat_g:.1f} g")
                if meal.note:
                    print(f"     Note     : {meal.note}")
            print()

        print("  " + "-" * 38)
        print(f"  📊 DAILY TOTAL    : {day.total_calories:.0f} kcal")
        print(f"     Daily Target   : {profile.daily_calories:.0f} kcal")

        food_deficit = max(0.0, profile.daily_calories - day.total_calories)
        if food_deficit > 0:
            print(f"     Food Deficit   : -{food_deficit:.0f} kcal")

        workout_recovery = 0.0
        if day.extra_workout_min > 0:
            kcal_burn        = day.extra_workout_min * BURN_RATE_KCAL_MIN
            workout_recovery = kcal_burn
            print(f"  🏃 EXERCISE       : +{day.extra_workout_min} min cardio")
            print(f"     Est. Burned    : ~{kcal_burn:.0f} kcal")

        day_recovery = food_deficit + workout_recovery
        cumulative  += day_recovery
        pct          = min(100.0, (cumulative / plan.calorie_surplus) * 100)
        filled       = int(pct / 5)
        bar          = "X" * filled + "." * (20 - filled)
        print(f"  📈 RECOVERY       : {cumulative:.0f} / {plan.calorie_surplus:.0f} kcal  "
              f"({pct:.1f}%)")
        print(f"     [{bar}]")

    print()
    print(SEP)
    if cumulative >= plan.calorie_surplus * 0.95:
        print(f"  ✅ Surplus fully compensated in {plan.window_days} day(s)!")
    else:
        shortfall = plan.calorie_surplus - cumulative
        print(f"  ⚠️  Partial recovery. ~{shortfall:.0f} kcal still remaining.")
        print("     Consider one extra light walk or an additional day.")
    print(SEP)
    print()


print("✅ print_meal_compensation_plan() ready.")
print("   Usage: print_meal_compensation_plan(plan, user)")
# ✅ Calculate total calories from all meals


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — CompensationPlannerService  (identical to original + _visualize)
# ══════════════════════════════════════════════════════════════════════════════

class CompensationPlannerService:
    """
    Orchestrator: detects cheat days, predicts strategy via ML,
    builds the compensation plan, and optionally visualises the output.
    """

    def __init__(self, model, scaler, le_goal, le_activity, le_label):
        self.model       = model
        self.scaler      = scaler
        self.le_goal     = le_goal
        self.le_activity = le_activity
        self.le_label    = le_label

    def _is_cheat_day(self, eaten_calories: float, profile: UserProfile) -> bool:
        surplus_pct = (eaten_calories - profile.daily_calories) / profile.daily_calories
        return surplus_pct >= CHEAT_THRESHOLD_PCT

    def _calculate_surplus(self, eaten_calories: float, profile: UserProfile) -> float:
        return max(0.0, eaten_calories - profile.daily_calories)

    def _predict_strategy(
        self, profile: UserProfile, surplus: float
    ) -> Tuple[StrategyType, float]:
        surplus_pct  = surplus / profile.daily_calories
        goal_enc     = self.le_goal.transform([profile.goal.value])[0]
        activity_enc = self.le_activity.transform([profile.activity_level.value])[0]

        feature_vector = np.array([[
            surplus_pct,
            surplus,
            goal_enc,
            activity_enc,
            profile.cheat_frequency,
            profile.age,
        ]])
        fv_s       = self.scaler.transform(feature_vector)
        pred_enc   = self.model.predict(fv_s)[0]
        confidence = self.model.predict_proba(fv_s)[0].max()
        strategy   = StrategyType(self.le_label.inverse_transform([pred_enc])[0])
        return strategy, confidence

    def _visualize(
        self,
        plan            : CompensationPlan,
        profile         : UserProfile,
        eaten_calories  : float,
    ):
        """4-panel chart: daily calories, macro stacks, workout minutes, cheat meal pie."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(
            f"Compensation Plan — {profile.name}  |  "
            f"Goal: {plan.goal.value}  |  "
            f"Surplus: {plan.calorie_surplus:.0f} kcal ({plan.severity.upper()})",
            fontsize=13, fontweight="bold", y=1.01
        )

        dates  = [d.date.strftime("%b %d") for d in plan.days]
        cals   = [d.total_calories for d in plan.days]
        target = [profile.daily_calories] * len(dates)

        # Panel 1 — Daily calorie plan
        ax = axes[0, 0]
        ax.plot(dates, target, "g--", linewidth=2, label="Daily Target", alpha=0.8)
        ax.bar(dates, cals, color=sns.color_palette("muted")[0], alpha=0.85, label="Plan")
        ax.axhline(MIN_DAILY_CALORIES, color="red", linestyle=":", linewidth=1.5,
                   label=f"Safety Floor ({MIN_DAILY_CALORIES} kcal)")
        ax.set_title("📅 Daily Calorie Plan", fontweight="bold")
        ax.set_ylabel("Calories (kcal)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9)

        # Panel 2 — Macro stacked bar
        ax = axes[0, 1]
        prots_cal = [sum(m.protein_g for m in d.meals) * 4 for d in plan.days]
        carbs_cal = [sum(m.carbs_g   for m in d.meals) * 4 for d in plan.days]
        fats_cal  = [sum(m.fat_g     for m in d.meals) * 9 for d in plan.days]
        x         = np.arange(len(dates))
        ax.bar(x, prots_cal, label="Protein", color="#4C72B0")
        ax.bar(x, carbs_cal, bottom=prots_cal, label="Carbs", color="#55A868")
        ax.bar(x, fats_cal,
               bottom=np.array(prots_cal) + np.array(carbs_cal),
               label="Fat", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=30)
        ax.set_title("🥗 Macro Distribution per Day", fontweight="bold")
        ax.set_ylabel("Calories (kcal equiv.)")
        ax.legend(fontsize=9)

        # Panel 3 — Extra workout minutes
        ax = axes[1, 0]
        workouts = [d.extra_workout_min for d in plan.days]
        colors   = ["#e74c3c" if w > 0 else "#bdc3c7" for w in workouts]
        bars     = ax.bar(dates, workouts, color=colors, alpha=0.85)
        for bar, val in zip(bars, workouts):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val}m", ha="center", fontsize=9)
        ax.set_title("🏃 Extra Workout (min/day)", fontweight="bold")
        ax.set_ylabel("Minutes")
        ax.tick_params(axis="x", rotation=30)

        # Panel 4 — Cheat meal calorie pie
        ax = axes[1, 1]
        if eaten_calories > 0:
            wedge_colors = sns.color_palette("Set2", 1)
            ax.pie([eaten_calories], labels=["Cheat Meal"], autopct="%1.0f%%",
                   colors=wedge_colors, startangle=90,
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
            ax.set_title(f"🍕 Cheat Meal ({eaten_calories:.0f} kcal total)",
                         fontweight="bold")

        plt.tight_layout()
        plt.savefig("compensation_plan.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  📊 Chart saved → compensation_plan.png")

    def process(
        self,
        profile          : UserProfile,
        eaten_calories   : float,
        cheat_date       : date,
        override_strategy: Optional[StrategyType] = None,
        visualize        : bool = False,
    ) -> Tuple[Optional[CompensationPlan], float, str]:
        """
        Main entry point.
        Returns (plan, surplus, severity). plan is None if no compensation needed.
        """
        if not self._is_cheat_day(eaten_calories, profile):
            pct = (eaten_calories / profile.daily_calories - 1) * 100
            print(f"  ✅ Normal day (+{pct:.1f}% over target). No compensation needed.")
            return None, 0.0, "none"

        surplus = self._calculate_surplus(eaten_calories, profile)
        print(f"\n  🍕 Cheat day detected!")
        print(f"     Ate      : {eaten_calories:.0f} kcal")
        print(f"     Target   : {profile.daily_calories:.0f} kcal")
        print(f"     Surplus  : {surplus:.0f} kcal  "
              f"[{_classify_severity(surplus, profile).upper()}]")

        if override_strategy:
            strategy, confidence = override_strategy, 1.0
            print(f"     Strategy : {strategy.value}  [manual override]")
        else:
            strategy, confidence = self._predict_strategy(profile, surplus)
            print(f"     Strategy : {strategy.value}  "
                  f"(ML confidence: {confidence*100:.1f}%)")

        plan               = STRATEGY_REGISTRY[strategy]().build_plan(
            profile, cheat_date, surplus, strategy
        )
        plan.ml_confidence = confidence
        print(f"     Window   : {plan.window_days} day(s)  <- auto-calculated from surplus")

        if visualize:
            self._visualize(plan, profile, eaten_calories)

        return plan, surplus, _classify_severity(surplus, profile)


planner = CompensationPlannerService(
    model       = rf,
    scaler      = scaler,
    le_goal     = le_goal,
    le_activity = le_activity,
    le_label    = le_label,
)
print("✅ CompensationPlannerService ready!")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — Dart field mapping helpers
# ══════════════════════════════════════════════════════════════════════════════

GOAL_MAP: Dict[str, Goal] = {
    "weight loss" : Goal.WEIGHT_LOSS,
    "weight gain" : Goal.MUSCLE_GAIN,
    "maintain"    : Goal.MAINTENANCE,
    "weight_loss" : Goal.WEIGHT_LOSS,
    "muscle_gain" : Goal.MUSCLE_GAIN,
    "maintenance" : Goal.MAINTENANCE,
}

ACTIVITY_MAP: Dict[str, ActivityLevel] = {
    "sedentary"         : ActivityLevel.SEDENTARY,
    "lightly active"    : ActivityLevel.LIGHT,
    "moderately active" : ActivityLevel.MODERATE,
    "very active"       : ActivityLevel.ACTIVE,
    "extra active"      : ActivityLevel.VERY_ACTIVE,
}

def map_goal(raw: str) -> Goal:
    key = raw.strip().lower()
    if key not in GOAL_MAP:
        raise ValueError(f"Unknown goal '{raw}'. Expected: {list(GOAL_MAP.keys())}")
    return GOAL_MAP[key]

def map_activity(raw: str) -> ActivityLevel:
    key = raw.strip().lower()
    if key not in ACTIVITY_MAP:
        raise ValueError(f"Unknown activitylevel '{raw}'. Expected: {list(ACTIVITY_MAP.keys())}")
    return ACTIVITY_MAP[key]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — FastAPI app + Pydantic models matching cheatmeal.dart exactly
# ══════════════════════════════════════════════════════════════════════════════



class UserPayload(BaseModel):
    """Mirrors _fetchUserProfile() in cheatmeal.dart exactly."""
    user_id               : str
    name                  : str
    age                   : str           # Firestore stores as string
    weight                : str           # "weight" — matches profile_setup.dart
    height                : str           # "height" — matches profile_setup.dart
    gender                : str
    goal                  : str           # "Weight Loss" | "Weight Gain" | "Maintain"
    activitylevel         : str           # "activitylevel" — matches profile_setup.dart
    bmi                   : float = 0.0
    bmr                   : float = 0.0
    tdee                  : float = 0.0
    daily_calories_target : float = 0.0
    daily_protein_target  : float = 0.0
    daily_carbs_target    : float = 0.0
    daily_fats_target     : float = 0.0


class CheatMealPayload(BaseModel):
    """Mirrors cheatMealForBackend map in cheatmeal.dart."""
    food_name  : str
    calories   : float
    protein_g  : float = 0.0
    carbs_g    : float = 0.0
    fat_g      : float = 0.0
    is_cheat   : bool  = True
    created_at : str   = ""


class CheatMealRequest(BaseModel):
    """Top-level payload — matches the payload dict in cheatmeal.dart."""
    user       : UserPayload
    cheat_meal : CheatMealPayload


class MealPlanResponse(BaseModel):
    meal_type         : str
    adjusted_calories : float
    protein_g         : float
    carbs_g           : float
    fat_g             : float
    note              : str = ""


class DayPlanResponse(BaseModel):
    date              : str
    meals             : List[MealPlanResponse]
    extra_workout_min : int
    day_note          : str


class CheatMealResponse(BaseModel):
    status            : str    # "compensation_needed" | "no_action_needed"
    user_id           : str
    food_name         : str
    calories_eaten    : float
    daily_target      : float
    surplus           : float
    severity          : str
    strategy          : str
    ml_confidence     : float
    window_days       : int
    compensation_days : List[DayPlanResponse]
    warnings          : List[str]
    goal_tip          : str    # per-goal recovery advice
    message           : str


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — /cheatmeal endpoint
# ══════════════════════════════════════════════════════════════════════════════

def cheatmeal_route(data: dict):
    try:
        user_data = data.get("user")
        cheat_meal = data.get("cheat_meal")


        if not user_data or not cheat_meal:
            return {"status": "error", "message": "Invalid payload"}

        # 1️⃣ Map enums
        goal = map_goal(user_data["goal"])
        activity_level = map_activity(user_data["activitylevel"])

        # 2️⃣ Parse numbers
        age = int(float(user_data["age"]))
        weight_kg = float(user_data["weight"])
        height_cm = float(user_data["height"])

        # 3️⃣ Build profile
        profile = UserProfile(
            user_id=user_data["user_id"],
            name=user_data["name"],
            age=age,
            weight_kg=weight_kg,
            height_cm=height_cm,
            gender=user_data["gender"],
            goal=goal,
            activity_level=activity_level,
            cheat_frequency=1,
            bmi=user_data.get("bmi", 0),
            bmr=user_data.get("bmr", 0),
            tdee=user_data.get("tdee", 0),
            daily_calories=user_data.get("daily_calories_target", 0),
            daily_protein_g=user_data.get("daily_protein_target", 0),
            daily_carbs_g=user_data.get("daily_carbs_target", 0),
            daily_fat_g=user_data.get("daily_fats_target", 0),
        )

        # ✅ STEP 4 — Calculate real daily intake
        all_meals = data.get("all_meals", [])

        if all_meals and isinstance(all_meals, list):
            eaten_calories = sum(
                float(m.get("calories", 0)) for m in all_meals
            )
        else:
            eaten_calories = float(cheat_meal.get("calories", 0))
        # ✅ STEP 5 — Run planner with REAL intake
        plan, surplus, severity = planner.process(
            profile=profile,
            eaten_calories=eaten_calories,
            cheat_date=date.today(),
            visualize=False,
        )
        tip = GOAL_TIP.get(goal, "")

        # ✅ No compensation needed
        if plan is None:
            return {
                "status": "no_action_needed",
                "message": "Within calorie target",
                "surplus": 0,
                "strategy": "none"
            }

        # ✅ Build response
        days = []
        for d in plan.days:
            days.append({
                "date": d.date.isoformat(),
                "meals": [
                    {
                        "meal_type": m.meal_type.value,
                        "calories": m.adjusted_calories,
                        "protein_g": m.protein_g,
                        "carbs_g": m.carbs_g,
                        "fat_g": m.fat_g,
                        "note": m.note
                    } for m in d.meals
                ],
                "extra_workout_min": d.extra_workout_min,
                "note": d.day_note
            })

        return {
            "status": "success",
            "surplus": surplus,
            "severity": severity,
            "strategy": plan.strategy_used.value,
            "window_days": plan.window_days,
            "compensation_days": days,
            "goal_tip": tip,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}