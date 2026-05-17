"""
F1 Lap Time Prediction Model
Trains a Gradient Boosting model across multiple 2024 races and visualises
feature importance, accuracy, error distribution, and tire degradation projections.
"""

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

fastf1.Cache.enable_cache("../cache")

# ── Races to train on ─────────────────────────────────────────────────────
RACES = ["Bahrain", "Saudi Arabia", "Australia", "Japan"]
YEAR  = 2024

# ── 1. Load & assemble training data ──────────────────────────────────────
print(f"\nLoading {len(RACES)} races to build training dataset...\n")

all_laps = []

for race_name in RACES:
    print(f"  → {race_name}...", end=" ", flush=True)
    try:
        session = fastf1.get_session(YEAR, race_name, "R")
        session.load(telemetry=False, messages=False)

        # Weather: resample to per-minute and forward-fill for merging
        weather = session.weather_data[["Time", "AirTemp", "TrackTemp"]].copy()
        weather = weather.set_index("Time").resample("1min").mean().ffill().reset_index()

        laps = session.laps.copy()
        laps = laps[laps["IsAccurate"] == True].copy()
        laps = laps[laps["LapTime"].notna()].copy()
        laps = laps[laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()

        # Merge nearest weather reading
        laps = pd.merge_asof(
            laps.sort_values("LapStartTime"),
            weather.rename(columns={"Time": "LapStartTime"}),
            on="LapStartTime",
            direction="nearest"
        )

        laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
        laps["Race"] = race_name

        # Fuel load proxy: drops ~1.6 kg/lap; use lap number scaled to max laps
        max_lap = laps["LapNumber"].max()
        laps["FuelLoad"] = (max_lap - laps["LapNumber"]) * 1.6   # kg remaining (relative)

        all_laps.append(laps[[
            "Race", "Driver", "LapTimeSec", "TyreLife", "Compound",
            "LapNumber", "FuelLoad", "AirTemp", "TrackTemp",
            "Stint", "FreshTyre"
        ]])
        print(f"{len(laps)} laps")

    except Exception as e:
        print(f"SKIPPED ({e})")

df = pd.concat(all_laps, ignore_index=True)
df = df.dropna()
print(f"\nTotal laps in dataset: {len(df)}")
print(f"Drivers: {df['Driver'].nunique()} | Races: {df['Race'].nunique()}")

# ── 2. Feature engineering ────────────────────────────────────────────────
compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
df["CompoundCode"] = df["Compound"].map(compound_map)
df["FreshTyreFlag"] = df["FreshTyre"].astype(int)

le_driver = LabelEncoder()
df["DriverCode"] = le_driver.fit_transform(df["Driver"])

# Normalize lap time per circuit: subtract each race's median clean lap time.
# This removes the 10–15s gap between circuits (Bahrain ~93s vs China ~97s)
# so the model learns degradation patterns, not "which track is this?"
df["RaceMedian"] = df.groupby("Race")["LapTimeSec"].transform("median")
df["LapTimeDelta"] = df["LapTimeSec"] - df["RaceMedian"]  # target: seconds above/below median

# Interaction: lets model learn SOFT degrades steeply, HARD stays flat
df["TyreLife_x_Compound"] = df["TyreLife"] * df["CompoundCode"]

FEATURES = [
    "TyreLife",              # laps on current tire set
    "CompoundCode",          # SOFT=0, MEDIUM=1, HARD=2
    "TyreLife_x_Compound",   # interaction: per-compound degradation curve
    "FuelLoad",              # kg remaining proxy (car weight)
    "LapNumber",             # race position
    "TrackTemp",             # surface temperature
    "AirTemp",               # ambient temperature
    "FreshTyreFlag",         # first lap on fresh rubber (out-lap flag)
    "DriverCode",            # driver pace/management style
]

X = df[FEATURES]
y = df["LapTimeDelta"]   # seconds above/below each circuit's median

# ── 3. Train / test split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Gradient Boosting model (target = delta from circuit median)...")
model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"  MAE  : {mae:.3f} sec  (mean absolute prediction error)")
print(f"  R²   : {r2:.4f}  (1.0 = perfect; now measures degradation, not circuit identity)")

# ── 4. What-if projection: tire compounds over a 30-lap stint ─────────────
STINT_LAPS   = 30
stint_ages   = np.arange(1, STINT_LAPS + 1)
base_lap_num = 20
base_track_T = df["TrackTemp"].mean()
base_air_T   = df["AirTemp"].mean()
driver_code  = 0    # same for all — so the comparison is compound vs compound only

projections = {}
for compound, code in compound_map.items():
    rows = pd.DataFrame({
        "TyreLife":            stint_ages,
        "CompoundCode":        code,
        "TyreLife_x_Compound": stint_ages * code,
        "FuelLoad":            (50 - stint_ages) * 1.6,
        "LapNumber":           base_lap_num + stint_ages,
        "TrackTemp":           base_track_T,
        "AirTemp":             base_air_T,
        "FreshTyreFlag":       (stint_ages == 1).astype(int),
        "DriverCode":          driver_code,
    })
    projections[compound] = model.predict(rows)

# ── 5. Plot ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    f"F1 Lap Time Prediction Model — {YEAR} ({', '.join(RACES)})",
    fontsize=16, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# Panel 1: Feature importance
ax1 = fig.add_subplot(gs[0, 0])
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
colors = ["#e74c3c" if v > 0.15 else "#3498db" for v in importances]
bars = ax1.barh(importances.index, importances.values, color=colors)
for bar, val in zip(bars, importances.values):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=8)
ax1.set_xlabel("Feature Importance (fraction of variance explained)")
ax1.set_title("What Predicts Lap Time?")
ax1.set_xlim(0, importances.max() * 1.25)
ax1.grid(axis="x", alpha=0.3)

# Panel 2: Actual vs Predicted (delta)
ax2 = fig.add_subplot(gs[0, 1])
sample = np.random.choice(len(y_test), min(600, len(y_test)), replace=False)
ax2.scatter(y_test.iloc[sample], y_pred[sample],
            alpha=0.35, s=12, color="#2980b9", edgecolors="none")
lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
ax2.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax2.axhline(0, color="gray", lw=0.8, alpha=0.5)
ax2.axvline(0, color="gray", lw=0.8, alpha=0.5)
ax2.set_xlabel("Actual Δ from circuit median (sec)")
ax2.set_ylabel("Predicted Δ from circuit median (sec)")
ax2.set_title(f"Actual vs Predicted (normalised)\nMAE = {mae:.3f}s  |  R² = {r2:.4f}")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Panel 3: Residual distribution
ax3 = fig.add_subplot(gs[1, 0])
residuals = y_test.values - y_pred
ax3.hist(residuals, bins=50, color="#2ecc71", edgecolor="white", linewidth=0.4)
ax3.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
ax3.axvline(residuals.mean(), color="orange", linestyle="--", linewidth=1.5,
            label=f"Mean error: {residuals.mean():.3f}s")
ax3.set_xlabel("Prediction Error (Actual − Predicted, seconds)")
ax3.set_ylabel("Count")
ax3.set_title("Error Distribution\n(errors now about degradation, not circuit gaps)")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Panel 4: Compound degradation projection
ax4 = fig.add_subplot(gs[1, 1])
compound_colors = {"SOFT": "#e74c3c", "MEDIUM": "#f39c12", "HARD": "#95a5a6"}
for compound, preds in projections.items():
    ax4.plot(stint_ages, preds, color=compound_colors[compound],
             lw=2.5, label=compound)
    ax4.fill_between(stint_ages, preds[0], preds,
                     color=compound_colors[compound], alpha=0.07)

# Annotate the warm-up zone
ax4.axvspan(1, 4, color="gray", alpha=0.08, zorder=0)
ax4.text(2.5, ax4.get_ylim()[0] if ax4.get_ylim()[0] > 0 else 90,
         "warm-up\nzone", ha="center", fontsize=7.5, color="gray",
         va="bottom")

# Annotate the cliff zone for soft
soft_preds = projections["SOFT"]
cliff_lap  = int(np.argmax(np.diff(soft_preds) > 0.12)) + 2   # first lap of steep rise
if 5 < cliff_lap < STINT_LAPS:
    ax4.annotate("SOFT cliff →",
                 xy=(cliff_lap, soft_preds[cliff_lap - 1]),
                 xytext=(cliff_lap + 3, soft_preds[cliff_lap - 1] + 0.5),
                 fontsize=8, color="#e74c3c",
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))

ax4.axhline(0, color="black", lw=1, linestyle="--", alpha=0.4, label="Circuit median pace")
ax4.set_xlabel("Laps into Stint (Tyre Life)")
ax4.set_ylabel("Seconds above/below circuit median")
ax4.set_title("Model Projection: Compound Degradation\n"
              "(0 = median pace; positive = slower; out-lap spike visible at lap 1)")
ax4.legend(title="Compound", fontsize=9)
ax4.grid(alpha=0.3)

plt.savefig("../outputs/f1_prediction.png", dpi=150, bbox_inches="tight")
print("\nChart saved: f1_prediction.png")
plt.show()

# ── 6. Interpret what the model learned ───────────────────────────────────
print(f"\n{'='*60}")
print("  WHAT THE MODEL LEARNED")
print(f"{'='*60}")
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
labels = {
    "TyreLife":            "Laps on tires — primary degradation signal",
    "FuelLoad":            "Car weight — heavier early = slower laps",
    "LapNumber":           "Position in race — correlated with fuel/safety cars",
    "TrackTemp":           "Surface temp — warmer = more grip (up to a point)",
    "CompoundCode":          "Tire type — SOFT vs HARD baseline pace",
    "TyreLife_x_Compound":   "Degradation rate differs by compound (interaction term)",
    "DriverCode":            "Driver pace and tire management style",
    "AirTemp":               "Ambient temp — affects tire operating window",
    "FreshTyreFlag":         "Out-lap / cold tyre — genuine slowdown at stint start",
}
for feat, score in imp.items():
    print(f"  {feat:<22} {score:.3f}   {labels.get(feat,'')}")

print(f"\n  Model predicts degradation delta within ±{mae:.2f} seconds on average")
print(f"  (Circuit baseline removed — model now measures tire behaviour only)\n")
