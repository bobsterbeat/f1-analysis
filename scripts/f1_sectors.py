"""
F1 Sector Time Analysis
Breaks lap times into S1, S2, S3 to show WHERE drivers gain/lose time.

Four panels:
  1. Driver sector ranking heatmap — who dominates which sector
  2. Top driver sector signatures — how each elite driver makes their time
  3. Sector variability — which sector is hardest to be consistent in
  4. Wet vs dry sector gains — where do wet-weather specialists actually gain?
"""

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
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

# ── 1. Load all cached races ───────────────────────────────────────────────
DRY_RACES = ["Bahrain","Saudi Arabia","Australia","Japan","China",
             "Monaco","Spain","Hungary","Netherlands","Italy",
             "Azerbaijan","Singapore","Abu Dhabi"]
WET_RACES = ["Canada","Britain","Brazil"]

print("Loading races for sector analysis...\n")

def load_sector_laps(races, is_wet_race=False):
    frames = []
    for race in races:
        try:
            s = fastf1.get_session(YEAR, race, "R")
            s.load(telemetry=False, messages=False)

            # Get team info
            teams = s.results[["Abbreviation","TeamName"]].set_index("Abbreviation")

            w = s.weather_data[["Time","Rainfall"]].copy()

            laps = s.laps.copy()
            laps = laps[(laps["IsAccurate"]==True) & laps["LapTime"].notna()].copy()
            laps = laps[["Driver","LapTime","Sector1Time","Sector2Time",
                         "Sector3Time","LapNumber","LapStartTime"]].copy()

            laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
            laps["S1"] = laps["Sector1Time"].dt.total_seconds()
            laps["S2"] = laps["Sector2Time"].dt.total_seconds()
            laps["S3"] = laps["Sector3Time"].dt.total_seconds()
            laps = laps.dropna(subset=["S1","S2","S3"])

            laps["Team"] = laps["Driver"].map(teams["TeamName"])
            laps["Race"] = race

            # Merge weather for wet lap flag
            laps = pd.merge_asof(laps.sort_values("LapStartTime"),
                                 w.rename(columns={"Time":"LapStartTime"}),
                                 on="LapStartTime", direction="nearest")
            laps["IsWet"] = laps["Rainfall"] > 0

            # Normalise each sector within its circuit
            for col in ["S1","S2","S3","LapTimeSec"]:
                laps[f"{col}_delta"] = laps[col] - laps[col].median()

            frames.append(laps[["Race","Driver","Team","IsWet",
                                 "S1","S2","S3","LapTimeSec",
                                 "S1_delta","S2_delta","S3_delta","LapTimeSec_delta"]])
            print(f"  {race}: {len(laps)} laps")
        except Exception as e:
            print(f"  {race}: skipped ({e})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

dry_laps = load_sector_laps(DRY_RACES, is_wet_race=False)
wet_laps  = load_sector_laps(WET_RACES, is_wet_race=True)
all_laps  = pd.concat([dry_laps, wet_laps], ignore_index=True)

print(f"\nTotal: {len(all_laps)} laps | {all_laps['Driver'].nunique()} drivers")

# ── 2. Per-driver median sector deltas (dry laps only) ────────────────────
dry_clean = all_laps[(all_laps["IsWet"]==False)].copy()

driver_sectors = (dry_clean.groupby("Driver")
                            .agg(S1=("S1_delta","median"),
                                 S2=("S2_delta","median"),
                                 S3=("S3_delta","median"),
                                 Overall=("LapTimeSec_delta","median"),
                                 Count=("S1","count"))
                            .reset_index()
                            .query("Count >= 80"))   # drivers with enough dry laps

# Overall rank by total pace
driver_sectors = driver_sectors.sort_values("Overall").reset_index(drop=True)

# Sector rank (lower delta = better rank)
for col in ["S1","S2","S3","Overall"]:
    driver_sectors[f"{col}_rank"] = driver_sectors[col].rank().astype(int)

# ── 3. Wet vs dry sector breakdown (key drivers) ──────────────────────────
# For drivers with wet laps, compute sector gains per sector
wet_clean = all_laps[all_laps["IsWet"]==True].copy()

def sector_medians(df, drivers):
    rows = []
    for drv in drivers:
        sub = df[df["Driver"]==drv]
        if len(sub) < 10: continue
        rows.append({"Driver": drv,
                     "S1": sub["S1_delta"].median(),
                     "S2": sub["S2_delta"].median(),
                     "S3": sub["S3_delta"].median()})
    return pd.DataFrame(rows).set_index("Driver")

# Wet-weather movers from previous analysis
WET_MOVERS = ["GAS","OCO","RUS","VER","LEC","NOR","PER","HAM","ALO"]
wet_movers = [d for d in WET_MOVERS if d in wet_clean["Driver"].unique()
              and len(wet_clean[wet_clean["Driver"]==d]) >= 15]

dry_sec  = sector_medians(dry_clean,  wet_movers)
wet_sec  = sector_medians(wet_clean,  wet_movers)
sec_gain = (wet_sec - dry_sec).dropna()   # negative = faster in wet vs their own dry pace

# ── 4. Sector consistency (std dev per sector) ────────────────────────────
consistency = (dry_clean.groupby("Driver")
                        .agg(S1_std=("S1","std"),
                             S2_std=("S2","std"),
                             S3_std=("S3","std"),
                             Count=("S1","count"))
                        .reset_index()
                        .query("Count >= 80"))

# ── PLOT ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle(f"F1 Sector Time Analysis — {YEAR} Season (Dry Races)",
             fontsize=16, fontweight="bold", color=NAVY, y=0.99)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel 1: Sector ranking heatmap ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
heat_data = driver_sectors.set_index("Driver")[["S1_rank","S2_rank","S3_rank","Overall_rank"]]
heat_data.columns = ["Sector 1","Sector 2","Sector 3","Overall"]

# Invert so rank 1 = best colour (green)
n = len(heat_data)
cmap = mcolors.LinearSegmentedColormap.from_list("rank", [GREEN, "#FFFDE7", RED])
im = ax1.imshow(heat_data.values, aspect="auto", cmap=cmap, vmin=1, vmax=n)

ax1.set_xticks(range(4))
ax1.set_xticklabels(heat_data.columns, fontsize=10, fontweight="bold")
ax1.set_yticks(range(n))
ax1.set_yticklabels(heat_data.index, fontsize=8.5)

for i in range(n):
    for j in range(4):
        val = int(heat_data.values[i, j])
        ax1.text(j, i, str(val), ha="center", va="center",
                 fontsize=8, color="white" if val <= 4 or val >= n-3 else "black",
                 fontweight="bold")

ax1.set_title("Sector Rank Heatmap (dry races)\n1 = fastest, green = strong", fontsize=11)
plt.colorbar(im, ax=ax1, label="Rank", shrink=0.8)

# ── Panel 2: Top 6 driver sector signatures ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
top6 = driver_sectors.head(8)["Driver"].tolist()
x = np.arange(3)
width = 0.10
sector_labels = ["Sector 1", "Sector 2", "Sector 3"]
colors_p2 = plt.cm.tab10(np.linspace(0, 0.8, len(top6)))

for i, (drv, color) in enumerate(zip(top6, colors_p2)):
    row = driver_sectors[driver_sectors["Driver"]==drv].iloc[0]
    vals = [row.S1, row.S2, row.S3]
    offset = (i - len(top6)/2) * width + width/2
    bars = ax2.bar(x + offset, vals, width, label=drv, color=color, alpha=0.85)

ax2.axhline(0, color="black", lw=1, alpha=0.5, linestyle="--")
ax2.set_xticks(x)
ax2.set_xticklabels(sector_labels, fontsize=11)
ax2.set_ylabel("Sector delta vs circuit median (sec)\nnegative = faster than median")
ax2.set_title("Sector Signatures — Top 8 Drivers\n(where they make their time)", fontsize=11)
ax2.legend(fontsize=8, ncol=2, loc="upper right")
ax2.grid(axis="y", alpha=0.3)
ax2.invert_yaxis()   # flip so faster (negative) = upward

# ── Panel 3: Sector consistency (std dev) ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
cons_top = consistency.sort_values("S1_std").head(16)
y3 = np.arange(len(cons_top))
w3 = 0.25

ax3.barh(y3 + w3,   cons_top["S1_std"], w3, color="#E74C3C", label="Sector 1", alpha=0.85)
ax3.barh(y3,        cons_top["S2_std"], w3, color="#F39C12", label="Sector 2", alpha=0.85)
ax3.barh(y3 - w3,   cons_top["S3_std"], w3, color="#7F8C8D", label="Sector 3", alpha=0.85)

ax3.set_yticks(y3)
ax3.set_yticklabels(cons_top["Driver"], fontsize=8.5)
ax3.set_xlabel("Lap time std dev per sector (sec) — lower = more consistent")
ax3.set_title("Sector Consistency — Top 16 Drivers\n(sorted by S1 consistency)", fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(axis="x", alpha=0.3)

# ── Panel 4: Wet weather sector gains ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
if len(sec_gain) > 0:
    x4 = np.arange(3)
    width4 = 0.08
    colors_p4 = plt.cm.Set2(np.linspace(0, 1, len(sec_gain)))
    for i, (drv, row) in enumerate(sec_gain.iterrows()):
        vals = [row.S1, row.S2, row.S3]
        offset = (i - len(sec_gain)/2) * width4 + width4/2
        color = GREEN if drv in ["GAS","OCO","RUS"] else (RED if drv in ["LEC","PER"] else MID)
        ax4.bar(x4 + offset, vals, width4, label=drv, color=colors_p4[i], alpha=0.85)

    ax4.axhline(0, color="black", lw=1.2, alpha=0.6, linestyle="--")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(["Sector 1","Sector 2","Sector 3"], fontsize=11)
    ax4.set_ylabel("Sector delta change: wet vs dry (sec)\nnegative = FASTER in wet relative to field")
    ax4.set_title("Wet Weather Sector Gains\n(negative = gains time vs field in the wet)", fontsize=11)
    ax4.legend(fontsize=8.5, ncol=2)
    ax4.grid(axis="y", alpha=0.3)
    ax4.invert_yaxis()   # flip so faster (negative) points upward

plt.savefig("../outputs/f1_sectors.png", dpi=150, bbox_inches="tight")
print("\nChart saved: outputs/f1_sectors.png")
plt.show()

# ── Print sector summary ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SECTOR RANKINGS — {YEAR} DRY RACES")
print(f"{'='*65}")
print(f"  {'Driver':<8} {'S1 rank':>8} {'S2 rank':>8} {'S3 rank':>8} {'Overall':>8}  Best sector")
print(f"  {'-'*58}")
for _, r in driver_sectors.iterrows():
    best = "S1" if r.S1_rank == min(r.S1_rank,r.S2_rank,r.S3_rank) else \
           "S2" if r.S2_rank == min(r.S1_rank,r.S2_rank,r.S3_rank) else "S3"
    print(f"  {r.Driver:<8} {r.S1_rank:>8} {r.S2_rank:>8} {r.S3_rank:>8} {r.Overall_rank:>8}  {best}")

print(f"\n  Wet weather sector analysis ({', '.join(wet_movers)}):")
print(f"  {'Driver':<8} {'S1 gain':>9} {'S2 gain':>9} {'S3 gain':>9}  Biggest wet gain")
print(f"  {'-'*52}")
for drv, row in sec_gain.sort_values("S1", ascending=False).iterrows():
    biggest = "S1" if row.S1==max(row.S1,row.S2,row.S3) else \
              "S2" if row.S2==max(row.S1,row.S2,row.S3) else "S3"
    print(f"  {drv:<8} {row.S1:>+9.3f} {row.S2:>+9.3f} {row.S3:>+9.3f}  {biggest}")
