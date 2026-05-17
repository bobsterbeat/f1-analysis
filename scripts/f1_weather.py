"""
F1 Weather Analysis
Four panels:
  1. Wet vs Dry driver ranking shift (same driver, different conditions)
  2. Track temperature vs lap time by compound (grip window)
  3. Brazil rain race — lap-by-lap transition from dry to wet
  4. Track temp range across all 2024 circuits
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
YEAR = 2024

NAVY  = "#1B3A6B"
WET   = "#2980B9"
DRY   = "#E67E22"
GREEN = "#1A7A4A"
RED   = "#C0392B"
MID   = "#6B7280"

# ── 1. Load dry races (already cached) ────────────────────────────────────
DRY_RACES = ["Bahrain","Saudi Arabia","Australia","Japan","China"]
WET_RACES = ["Canada","Britain","Brazil"]

print("Loading dry races...")
dry_laps_list = []
for race in DRY_RACES:
    s = fastf1.get_session(YEAR, race, "R")
    s.load(telemetry=False, messages=False)
    laps = s.laps.copy()
    laps = laps[(laps["IsAccurate"]==True) & laps["LapTime"].notna()].copy()
    laps = laps[laps["Compound"].isin(["SOFT","MEDIUM","HARD","INTERMEDIATE","WET"])].copy()
    w = s.weather_data[["Time","TrackTemp","AirTemp","Rainfall"]].copy()
    laps = pd.merge_asof(laps.sort_values("LapStartTime"),
                         w.rename(columns={"Time":"LapStartTime"}),
                         on="LapStartTime", direction="nearest")
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps["Race"] = race
    laps["IsWet"] = False
    dry_laps_list.append(laps[["Race","Driver","Team","LapTimeSec","TrackTemp",
                                "AirTemp","Rainfall","Compound","LapNumber","IsWet"]])
    print(f"  {race}: {len(laps)} laps")

print("\nLoading wet races...")
wet_laps_list = []
for race in WET_RACES:
    s = fastf1.get_session(YEAR, race, "R")
    s.load(telemetry=False, messages=False)
    laps = s.laps.copy()
    laps = laps[(laps["IsAccurate"]==True) & laps["LapTime"].notna()].copy()
    laps = laps[laps["Compound"].isin(["SOFT","MEDIUM","HARD","INTERMEDIATE","WET"])].copy()
    w = s.weather_data[["Time","TrackTemp","AirTemp","Rainfall"]].copy()
    laps = pd.merge_asof(laps.sort_values("LapStartTime"),
                         w.rename(columns={"Time":"LapStartTime"}),
                         on="LapStartTime", direction="nearest")
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps["Race"] = race
    laps["IsWet"] = laps["Rainfall"] > 0
    wet_laps_list.append(laps[["Race","Driver","Team","LapTimeSec","TrackTemp",
                                "AirTemp","Rainfall","Compound","LapNumber","IsWet"]])
    print(f"  {race}: {len(laps)} laps  (wet laps: {laps['IsWet'].sum()})")

dry_df = pd.concat(dry_laps_list, ignore_index=True)
wet_df = pd.concat(wet_laps_list, ignore_index=True)
all_df = pd.concat([dry_df, wet_df], ignore_index=True)

# ── 2. Wet vs Dry driver ranking ───────────────────────────────────────────
# Dry rank: median normalised lap time across dry races
# Wet rank: median normalised lap time across wet-flagged laps only

def normalised_median(df, condition_mask):
    sub = df[condition_mask].copy()
    results = []
    for (race, driver), g in sub.groupby(["Race","Driver"]):
        circuit_med = df[df["Race"]==race]["LapTimeSec"].median()
        delta = g["LapTimeSec"].median() - circuit_med
        results.append({"Driver": driver, "Delta": delta, "Count": len(g)})
    res = pd.DataFrame(results)
    return (res[res["Count"] >= 5]
              .groupby("Driver")["Delta"]
              .median()
              .sort_values())

dry_rank = normalised_median(all_df, all_df["IsWet"]==False)
wet_rank  = normalised_median(all_df, all_df["IsWet"]==True)

# Drivers in both conditions with enough laps
common = sorted(set(dry_rank.index) & set(wet_rank.index))
dry_pos = {d: i+1 for i, d in enumerate(dry_rank.reindex(common).sort_values().index)}
wet_pos = {d: i+1 for i, d in enumerate(wet_rank.reindex(common).sort_values().index)}
ranking_df = pd.DataFrame({
    "Driver": common,
    "DryRank": [dry_pos[d] for d in common],
    "WetRank":  [wet_pos[d]  for d in common],
}).assign(RankChange=lambda x: x["DryRank"] - x["WetRank"])
ranking_df = ranking_df.sort_values("WetRank").reset_index(drop=True)

# ── 3. Track temp vs lap time (dry laps, slick compounds only) ────────────
slick_dry = dry_df[dry_df["Compound"].isin(["SOFT","MEDIUM","HARD"])].copy()
# Normalise lap times within each race
for race in DRY_RACES:
    mask = slick_dry["Race"] == race
    slick_dry.loc[mask, "LapDelta"] = (
        slick_dry.loc[mask, "LapTimeSec"] - slick_dry.loc[mask, "LapTimeSec"].median()
    )
# Bin track temps
slick_dry["TempBin"] = pd.cut(slick_dry["TrackTemp"], bins=range(15,60,3),
                               labels=[f"{t}–{t+3}" for t in range(15,57,3)])
temp_trend = (slick_dry.groupby(["TempBin","Compound"])["LapDelta"]
                        .agg(Median="median", Count="count")
                        .reset_index()
                        .dropna())

# ── 4. Brazil race — lap-by-lap transition ────────────────────────────────
brazil = fastf1.get_session(YEAR, "Brazil", "R")
brazil.load(telemetry=False, messages=False)
bz_laps = brazil.laps.copy()
# IsAccurate filters out safety car laps and laps under yellow — real racing pace only
bz_laps = bz_laps[(bz_laps["LapTime"].notna()) & (bz_laps["IsAccurate"]==True)].copy()
bz_laps["LapTimeSec"] = bz_laps["LapTime"].dt.total_seconds()
bz_w = brazil.weather_data[["Time","TrackTemp","Rainfall"]].copy()
bz_laps = pd.merge_asof(bz_laps.sort_values("LapStartTime"),
                         bz_w.rename(columns={"Time":"LapStartTime"}),
                         on="LapStartTime", direction="nearest")
# Field median per lap (racing laps only — safety car laps excluded)
bz_med = (bz_laps
           .groupby("LapNumber")
           .agg(MedianSec=("LapTimeSec","median"),
                Rain=("Rainfall","max"),
                TrackTemp=("TrackTemp","mean"))
           .reset_index())

# ── 5. Circuit track temp range (all 2024 races) ──────────────────────────
circuit_temps = [
    ("Bahrain",       21.9, 26.5),("Saudi Arabia",  29.1, 33.6),
    ("Australia",     36.7, 39.6),("Japan",          31.3, 40.5),
    ("China",         26.8, 32.3),("Miami",          39.3, 49.1),
    ("Monaco",        40.2, 50.0),("Canada",         19.9, 28.5),
    ("Spain",         38.3, 43.6),("Austria",        43.4, 49.0),
    ("Britain",       20.7, 37.9),("Hungary",        40.4, 49.7),
    ("Belgium",       37.7, 44.1),("Netherlands",    27.5, 32.4),
    ("Italy",         43.5, 54.6),("Azerbaijan",     39.3, 47.9),
    ("Singapore",     34.4, 38.5),("United States",  44.1, 48.0),
    ("Mexico",        30.6, 40.2),("Brazil",         23.2, 29.5),
    ("Las Vegas",     16.6, 19.9),("Qatar",          21.9, 23.8),
    ("Abu Dhabi",     29.1, 37.2),
]
ct = pd.DataFrame(circuit_temps, columns=["Circuit","TMin","TMax"])
ct["TMid"]  = (ct["TMin"] + ct["TMax"]) / 2
ct["Range"] = ct["TMax"] - ct["TMin"]
ct = ct.sort_values("TMid").reset_index(drop=True)

# ── PLOT ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle(f"F1 Weather Analysis — {YEAR} Season", fontsize=16,
             fontweight="bold", color=NAVY, y=0.99)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.36)

# ── Panel 1: Wet vs Dry ranking shift ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
y = np.arange(len(ranking_df))
for i, row in ranking_df.iterrows():
    color = GREEN if row.RankChange > 0 else (RED if row.RankChange < 0 else MID)
    ax1.plot([0, 1], [row.DryRank, row.WetRank], color=color, lw=1.8, alpha=0.8)
    ax1.scatter(0, row.DryRank, color=DRY, s=55, zorder=5)
    ax1.scatter(1, row.WetRank,  color=WET,  s=55, zorder=5)
    ax1.text(-0.08, row.DryRank, row.Driver, ha="right", va="center", fontsize=7.5)
    ax1.text(1.08,  row.WetRank,  row.Driver, ha="left",  va="center", fontsize=7.5)
    if abs(row.RankChange) >= 3:
        mid_x, mid_y = 0.5, (row.DryRank + row.WetRank) / 2
        sign = f"▲{row.RankChange:+.0f}" if row.RankChange > 0 else f"▼{row.RankChange:+.0f}"
        ax1.text(mid_x, mid_y, sign, ha="center", va="center",
                 fontsize=7, color=color, fontweight="bold")

ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Dry races", "Wet laps"], fontsize=11, fontweight="bold")
ax1.set_ylabel("Pace rank  (1 = fastest)")
ax1.invert_yaxis()
ax1.set_title("Wet vs Dry Driver Ranking\n(green = improves in rain, red = worse)", fontsize=11)
ax1.set_xlim(-0.5, 1.5)
ax1.grid(axis="y", alpha=0.2)
green_p = mpatches.Patch(color=GREEN, label="Improves in wet")
red_p   = mpatches.Patch(color=RED,   label="Worse in wet")
ax1.legend(handles=[green_p, red_p], fontsize=8.5, loc="lower right")

# ── Panel 2: Track temp vs lap time by compound ───────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
comp_colors = {"SOFT": "#E74C3C", "MEDIUM": "#F39C12", "HARD": "#7F8C8D"}
for compound, grp in temp_trend.groupby("Compound"):
    valid = grp[grp["Count"] >= 10].copy()
    if len(valid) < 3: continue
    ax2.plot(range(len(valid)), valid["Median"], color=comp_colors[compound],
             lw=2.2, label=compound, marker="o", ms=5)
    ax2.fill_between(range(len(valid)),
                     valid["Median"] - 0.3, valid["Median"] + 0.3,
                     color=comp_colors[compound], alpha=0.12)
    xticklabels = valid["TempBin"].tolist()

ax2.set_xticks(range(len(xticklabels)))
ax2.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7.5)
ax2.axhline(0, color="black", lw=0.8, alpha=0.4, linestyle="--")
ax2.set_ylabel("Lap time vs circuit median (sec)")
ax2.set_xlabel("Track surface temperature (°C)")
ax2.set_title("Tyre Performance Window by Track Temperature\n(0 = circuit median pace)", fontsize=11)
ax2.legend(title="Compound", fontsize=9)
ax2.grid(alpha=0.25)

# ── Panel 3: Brazil race — rain transition ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3b = ax3.twinx()

# Shade rain periods
for _, row in bz_med.iterrows():
    if row.Rain > 0:
        ax3.axvspan(row.LapNumber - 0.5, row.LapNumber + 0.5,
                    alpha=0.15, color=WET, zorder=0)

ax3.plot(bz_med["LapNumber"], bz_med["MedianSec"],
         color=NAVY, lw=2, label="Field median lap time", zorder=3)
ax3b.plot(bz_med["LapNumber"], bz_med["TrackTemp"],
          color=DRY, lw=1.4, linestyle="--", alpha=0.8, label="Track temp")

ax3.set_xlabel("Lap Number")
ax3.set_ylabel("Median lap time (seconds)", color=NAVY)
ax3b.set_ylabel("Track temperature (°C)", color=DRY)
ax3.set_title("2024 Brazilian GP — Rain Transition\n(racing laps only — safety car laps excluded)", fontsize=11)

rain_patch = mpatches.Patch(color=WET, alpha=0.3, label="Rainfall")
lap_line   = mpatches.Patch(color=NAVY, label="Lap time")
temp_line  = mpatches.Patch(color=DRY,  label="Track temp")
ax3.legend(handles=[rain_patch, lap_line, temp_line], fontsize=8.5, loc="upper left")
ax3.grid(alpha=0.25)

# ── Panel 4: Circuit track temperature range ──────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
y4 = np.arange(len(ct))
bar_colors = [WET if c in ["Canada","Britain","Brazil"] else
              ("#E8A838" if mid > 45 else NAVY)
              for c, mid in zip(ct["Circuit"], ct["TMid"])]

ax4.barh(y4, ct["TMax"] - ct["TMin"], left=ct["TMin"],
         color=bar_colors, height=0.6, alpha=0.85)
ax4.scatter(ct["TMid"], y4, color="white", s=20, zorder=5)

ax4.set_yticks(y4)
ax4.set_yticklabels(ct["Circuit"], fontsize=8.5)
ax4.set_xlabel("Track surface temperature (°C)")
ax4.set_title("Track Temperature Range — All 2024 Circuits\n(bar = min to max during race)", fontsize=11)

# Annotations
ax4.axvline(40, color=RED, lw=1.2, linestyle="--", alpha=0.6, label="40°C — tyre stress zone")
ax4.axvline(25, color=WET, lw=1.2, linestyle="--", alpha=0.6, label="25°C — warm-up struggles")
ax4.legend(fontsize=8.5)

wet_p  = mpatches.Patch(color=WET,  alpha=0.85, label="Wet race")
hot_p  = mpatches.Patch(color="#E8A838", alpha=0.85, label="Hot circuit (>45°C)")
dry_p  = mpatches.Patch(color=NAVY, alpha=0.85, label="Normal dry race")
ax4.legend(handles=[wet_p, hot_p, dry_p], fontsize=8.5, loc="lower right")
ax4.grid(axis="x", alpha=0.3)

plt.savefig("../outputs/f1_weather.png", dpi=150, bbox_inches="tight")
print("\nChart saved: outputs/f1_weather.png")
plt.show()

# ── Print wet vs dry ranking table ───────────────────────────────────────
print(f"\n{'='*58}")
print(f"  WET vs DRY DRIVER RANKING SHIFT — {YEAR}")
print(f"{'='*58}")
print(f"  {'Driver':<8} {'Dry':>5} {'Wet':>5} {'Change':>8}  Note")
print(f"  {'-'*52}")
for _, r in ranking_df.sort_values("RankChange", ascending=False).iterrows():
    chg = r.RankChange
    arrow = f"▲ +{chg:.0f}" if chg > 0 else (f"▼ {chg:.0f}" if chg < 0 else "  —")
    note = "Better in wet" if chg >= 3 else ("Worse in wet" if chg <= -3 else "Similar")
    print(f"  {r.Driver:<8} {r.DryRank:>5.0f} {r.WetRank:>5.0f} {arrow:>8}  {note}")
