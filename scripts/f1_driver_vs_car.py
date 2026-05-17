"""
F1 Driver vs Car Analysis
Separates driver contribution from car performance using teammate comparisons.

Logic: teammates share the same car. Any consistent lap time gap between them
is driver skill/style, not machinery. Car pace = average of both teammates.
Driver contribution = individual pace minus car baseline.
"""

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

fastf1.Cache.enable_cache("../cache")

RACES = ["Bahrain", "Saudi Arabia", "Australia", "Japan"]
YEAR  = 2024

# ── 1. Load clean lap data ─────────────────────────────────────────────────
print(f"\nLoading {len(RACES)} races...\n")

all_laps = []
for race_name in RACES:
    print(f"  → {race_name}...", end=" ", flush=True)
    session = fastf1.get_session(YEAR, race_name, "R")
    session.load(telemetry=False, messages=False)

    # Team assignments from session results
    teams = session.results[["Abbreviation", "TeamName"]].set_index("Abbreviation")

    laps = session.laps.copy()
    laps = laps[laps["IsAccurate"] == True].copy()
    laps = laps[laps["LapTime"].notna()].copy()
    laps = laps[laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps["Race"] = race_name
    laps["Team"] = laps["Driver"].map(teams["TeamName"])

    # Circuit median for normalisation
    circuit_median = laps["LapTimeSec"].median()
    laps["LapTimeDelta"] = laps["LapTimeSec"] - circuit_median

    all_laps.append(laps[["Race", "Driver", "Team", "LapTimeSec", "LapTimeDelta",
                           "LapNumber", "TyreLife", "Compound", "Stint"]])
    print(f"{len(laps)} laps")

df = pd.concat(all_laps, ignore_index=True)
print(f"\nTotal: {len(df)} laps | {df['Driver'].nunique()} drivers | {df['Team'].nunique()} teams")

# ── 2. Per-driver summary stats ────────────────────────────────────────────
# Median and std dev across all races (normalised so circuits are comparable)
driver_stats = (df.groupby(["Driver", "Team"])["LapTimeDelta"]
                  .agg(MedianDelta="median", Consistency="std", LapCount="count")
                  .reset_index())

# Only keep drivers with enough laps for meaningful stats
driver_stats = driver_stats[driver_stats["LapCount"] >= 30].copy()

# ── 3. Teammate gap decomposition ─────────────────────────────────────────
# For each team: identify the two drivers, compute gap and car baseline
teammate_pairs = []
for team, group in driver_stats.groupby("Team"):
    if len(group) < 2:
        continue
    drivers = group.sort_values("MedianDelta").reset_index(drop=True)
    fast_driver  = drivers.iloc[0]["Driver"]
    slow_driver  = drivers.iloc[1]["Driver"]
    fast_median  = drivers.iloc[0]["MedianDelta"]
    slow_median  = drivers.iloc[1]["MedianDelta"]
    car_baseline = (fast_median + slow_median) / 2   # car pace = teammate average
    gap          = slow_median - fast_median          # positive = faster driver wins

    teammate_pairs.append({
        "Team":         team,
        "FastDriver":   fast_driver,
        "SlowDriver":   slow_driver,
        "FastMedian":   fast_median,
        "SlowMedian":   slow_median,
        "CarBaseline":  car_baseline,
        "DriverGap":    gap,
        "FastConsist":  drivers.iloc[0]["Consistency"],
        "SlowConsist":  drivers.iloc[1]["Consistency"],
    })

pairs = pd.DataFrame(teammate_pairs).sort_values("CarBaseline")

# ── 4. Variance decomposition ─────────────────────────────────────────────
# Total variance in lap times = between-team (car) + within-team (driver)
grand_mean = driver_stats["MedianDelta"].mean()

# Between-team variance: how spread are car baselines?
car_baselines = pairs["CarBaseline"].values
between_var = np.var(car_baselines)

# Within-team variance: average squared gap between teammates / 4
within_var = np.mean((pairs["DriverGap"].values / 2) ** 2)

total_var = between_var + within_var
car_pct    = between_var / total_var * 100
driver_pct = within_var  / total_var * 100

print(f"\nVariance decomposition:")
print(f"  Car contribution  : {car_pct:.1f}%")
print(f"  Driver contribution: {driver_pct:.1f}%")

# ── 5. Plot ───────────────────────────────────────────────────────────────
NAVY  = "#1B3A6B"
RED   = "#C0392B"
GREEN = "#1A7A4A"
AMBER = "#D97706"
GREY  = "#6B7280"

fig = plt.figure(figsize=(18, 13))
fig.suptitle(f"Driver vs Car — {YEAR} F1 Season ({', '.join(RACES)})",
             fontsize=16, fontweight="bold", color=NAVY, y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.38)

# ── Panel 1: Teammate pace gap per team ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
pairs_sorted = pairs.sort_values("DriverGap", ascending=True)
y_pos = np.arange(len(pairs_sorted))

bars = ax1.barh(y_pos, pairs_sorted["DriverGap"], color=NAVY, height=0.55, zorder=3)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{row.FastDriver}  vs  {row.SlowDriver}"
                     for _, row in pairs_sorted.iterrows()], fontsize=9)
ax1.set_xlabel("Pace gap (sec) — faster teammate vs slower teammate")
ax1.set_title("Teammate Pace Gap\n(same car, same race — gap = driver skill)", fontsize=11)
ax1.axvline(pairs_sorted["DriverGap"].median(), color=AMBER, linestyle="--",
            lw=1.5, label=f"Median gap: {pairs_sorted['DriverGap'].median():.2f}s")
for bar, val in zip(bars, pairs_sorted["DriverGap"]):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f"+{val:.2f}s", va="center", fontsize=8.5, color=NAVY, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(axis="x", alpha=0.3)
ax1.set_xlim(0, pairs_sorted["DriverGap"].max() * 1.35)

# ── Panel 2: Car pace ranking (baseline) ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
pairs_car = pairs.sort_values("CarBaseline")
y_pos2 = np.arange(len(pairs_car))

colors2 = [GREEN if v <= pairs_car["CarBaseline"].median() else RED
           for v in pairs_car["CarBaseline"]]
ax2.barh(y_pos2, pairs_car["CarBaseline"], color=colors2, height=0.55, zorder=3)
ax2.set_yticks(y_pos2)

# Shorten team names for display
def short_team(name):
    return {"Red Bull Racing": "Red Bull", "Aston Martin": "Aston Martin",
            "Kick Sauber": "Sauber", "Haas F1 Team": "Haas"}.get(name, name)

ax2.set_yticklabels([short_team(t) for t in pairs_car["Team"]], fontsize=9)
ax2.axvline(0, color="black", lw=1, alpha=0.5)
ax2.set_xlabel("Seconds vs field median  (negative = faster)")
ax2.set_title("Car Pace Ranking\n(teammate average — driver effects cancelled out)", fontsize=11)
for i, (_, row) in enumerate(pairs_car.iterrows()):
    sign = "+" if row.CarBaseline >= 0 else ""
    ax2.text(row.CarBaseline + (0.02 if row.CarBaseline >= 0 else -0.02),
             i, f"{sign}{row.CarBaseline:.2f}s",
             va="center", ha="left" if row.CarBaseline >= 0 else "right",
             fontsize=8.5, fontweight="bold",
             color=GREEN if row.CarBaseline < 0 else RED)
ax2.grid(axis="x", alpha=0.3)

green_patch = mpatches.Patch(color=GREEN, label="Faster than field median")
red_patch   = mpatches.Patch(color=RED,   label="Slower than field median")
ax2.legend(handles=[green_patch, red_patch], fontsize=8.5)

# ── Panel 3: Pace vs Consistency scatter ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for _, row in driver_stats.iterrows():
    ax3.scatter(row["MedianDelta"], row["Consistency"],
                s=90, zorder=3,
                color=NAVY if row["MedianDelta"] <= driver_stats["MedianDelta"].median() else GREY)
    ax3.annotate(row["Driver"],
                 (row["MedianDelta"], row["Consistency"]),
                 textcoords="offset points", xytext=(6, 2), fontsize=7.5, color="#333333")

# Quadrant lines
med_pace    = driver_stats["MedianDelta"].median()
med_consist = driver_stats["Consistency"].median()
ax3.axvline(med_pace,    color=AMBER, lw=1.2, linestyle="--", alpha=0.7)
ax3.axhline(med_consist, color=AMBER, lw=1.2, linestyle="--", alpha=0.7)

# Quadrant labels
ax3_xlim = ax3.get_xlim()
ax3_ylim = ax3.get_ylim()
ax3.text(med_pace - 0.05, med_consist + 0.02, "Fast & Inconsistent",
         ha="right", fontsize=7.5, color=RED, style="italic")
ax3.text(med_pace + 0.05, med_consist + 0.02, "Slow & Inconsistent",
         ha="left", fontsize=7.5, color=GREY, style="italic")
ax3.text(med_pace - 0.05, med_consist - 0.02, "Fast & Consistent  ★",
         ha="right", fontsize=7.5, color=GREEN, style="italic")
ax3.text(med_pace + 0.05, med_consist - 0.02, "Slow & Consistent",
         ha="left", fontsize=7.5, color=GREY, style="italic")

ax3.set_xlabel("Median lap time vs field (sec) — left = faster")
ax3.set_ylabel("Lap time std dev (sec) — lower = more consistent")
ax3.set_title("Pace vs Consistency\n(ideal driver: bottom-left)", fontsize=11)
ax3.invert_xaxis()
ax3.grid(alpha=0.25)

# ── Panel 4: Variance decomposition pie ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
wedge_colors = [NAVY, "#5B8DB8"]
wedges, texts, autotexts = ax4.pie(
    [car_pct, driver_pct],
    labels=["Car\n(between teams)", "Driver\n(within teams)"],
    colors=wedge_colors,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.65,
    textprops={"fontsize": 11, "fontweight": "bold"},
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for at in autotexts:
    at.set_color("white")
    at.set_fontsize(13)
    at.set_fontweight("bold")

ax4.set_title("What Drives Lap Time Variation?\nCar vs Driver Decomposition", fontsize=11)

# Annotation box
summary_text = (
    f"Car accounts for {car_pct:.0f}% of lap time variation.\n"
    f"Drivers account for {driver_pct:.0f}%.\n\n"
    f"Largest teammate gap: {pairs['DriverGap'].max():.2f}s\n"
    f"  ({pairs.loc[pairs['DriverGap'].idxmax(), 'FastDriver']} vs "
    f"{pairs.loc[pairs['DriverGap'].idxmax(), 'SlowDriver']})\n\n"
    f"Smallest teammate gap: {pairs['DriverGap'].min():.2f}s\n"
    f"  ({pairs.loc[pairs['DriverGap'].idxmin(), 'FastDriver']} vs "
    f"{pairs.loc[pairs['DriverGap'].idxmin(), 'SlowDriver']})"
)
ax4.text(0, -1.55, summary_text, ha="center", va="top", fontsize=8.5,
         color="#333333", linespacing=1.6,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#F2F4F7", edgecolor="#D1D5DB"))

plt.savefig("../outputs/f1_driver_vs_car.png", dpi=150, bbox_inches="tight")
print("\nChart saved: outputs/f1_driver_vs_car.png")
plt.show()

# ── 6. Print summary table ────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TEAMMATE GAP SUMMARY — {YEAR} ({', '.join(RACES)})")
print(f"{'='*70}")
print(f"  {'Team':<22} {'Faster':>6} {'Slower':>6} {'Gap':>7}  {'Car baseline':>13}")
print(f"  {'-'*65}")
for _, row in pairs.sort_values("DriverGap", ascending=False).iterrows():
    print(f"  {short_team(row.Team):<22} {row.FastDriver:>6} {row.SlowDriver:>6} "
          f"  {row.DriverGap:>+.3f}s  {row.CarBaseline:>+.3f}s vs median")

print(f"\n  Car vs Driver split: {car_pct:.0f}% car  /  {driver_pct:.0f}% driver")
print(f"  (based on variance decomposition across {len(pairs)} teammate pairs)\n")
