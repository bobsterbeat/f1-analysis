"""
F1 Telemetry Explorer — FastF1 Intro Project
Shows what data is available and plots 4 graphs from a real race.
"""

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

# ── Setup ──────────────────────────────────────────────────────────────────
fastf1.Cache.enable_cache("../cache")          # saves downloaded data locally
fastf1.plotting.setup_mpl()

YEAR   = 2024
RACE   = "Bahrain"                          # change to any GP name
SESSION = "R"                               # R=Race, Q=Qualifying, FP1/FP2/FP3

print(f"\n{'='*55}")
print(f"  Loading {YEAR} {RACE} Grand Prix — session: {SESSION}")
print(f"{'='*55}\n")

session = fastf1.get_session(YEAR, RACE, SESSION)
session.load()

# ── PART 1: What's in the database? ───────────────────────────────────────
print("DRIVERS IN THIS SESSION:")
print(session.results[["Abbreviation","FullName","TeamName","Position"]].to_string(index=False))

print("\nLAP DATA COLUMNS AVAILABLE:")
print(list(session.laps.columns))

print("\nTELEMETRY CHANNELS (per lap):")
sample_lap = session.laps.pick_driver("HAM").iloc[0]
tel = sample_lap.get_telemetry()
print(list(tel.columns))
print(f"\nTelemetry points per lap sample: {len(tel)} rows")

print("\nSESSION WEATHER SNAPSHOT:")
print(session.weather_data[["Time","AirTemp","TrackTemp","Humidity","WindSpeed"]].head(5).to_string(index=False))

# ── PART 2: Pick drivers to compare ───────────────────────────────────────
DRIVER_A = "VER"
DRIVER_B = "HAM"

laps_a = session.laps.pick_drivers(DRIVER_A).pick_quicklaps().reset_index(drop=True)
laps_b = session.laps.pick_drivers(DRIVER_B).pick_quicklaps().reset_index(drop=True)

color_a = fastf1.plotting.get_driver_color(DRIVER_A, session)
color_b = fastf1.plotting.get_driver_color(DRIVER_B, session)

# ── FIGURE: 4-panel overview ───────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle(f"F1 Data Overview — {YEAR} {RACE} GP", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Panel 1: Lap times across the race ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(laps_a["LapNumber"], laps_a["LapTime"].dt.total_seconds(),
            color=color_a, s=20, label=DRIVER_A, zorder=3)
ax1.scatter(laps_b["LapNumber"], laps_b["LapTime"].dt.total_seconds(),
            color=color_b, s=20, label=DRIVER_B, zorder=3)
ax1.set_xlabel("Lap Number")
ax1.set_ylabel("Lap Time (seconds)")
ax1.set_title("Lap Times — Full Race")
ax1.legend()
ax1.grid(alpha=0.3)

# ── Panel 2: Tire degradation per stint ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for driver, laps, color in [(DRIVER_A, laps_a, color_a), (DRIVER_B, laps_b, color_b)]:
    for stint_num, stint in laps.groupby("Stint"):
        compound = stint["Compound"].iloc[0]
        stint_lap = stint["LapNumber"] - stint["LapNumber"].min()
        lap_sec   = stint["LapTime"].dt.total_seconds()
        ax2.plot(stint_lap, lap_sec, color=color, alpha=0.85,
                 label=f"{driver} — {compound} (stint {int(stint_num)})")

ax2.set_xlabel("Laps into Stint")
ax2.set_ylabel("Lap Time (seconds)")
ax2.set_title("Tire Degradation by Stint")
ax2.legend(fontsize=7, loc="upper left")
ax2.grid(alpha=0.3)

# ── Panel 3: Speed trace on fastest lap ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
fastest_a = session.laps.pick_drivers(DRIVER_A).pick_fastest()
fastest_b = session.laps.pick_drivers(DRIVER_B).pick_fastest()
tel_a = fastest_a.get_telemetry().add_distance()
tel_b = fastest_b.get_telemetry().add_distance()

ax3.plot(tel_a["Distance"], tel_a["Speed"], color=color_a, label=DRIVER_A, lw=1.2)
ax3.plot(tel_b["Distance"], tel_b["Speed"], color=color_b, label=DRIVER_B, lw=1.2, alpha=0.8)
ax3.set_xlabel("Distance (m)")
ax3.set_ylabel("Speed (km/h)")
ax3.set_title("Speed Trace — Fastest Lap Comparison")
ax3.legend()
ax3.grid(alpha=0.3)

# ── Panel 4: Throttle vs Brake on fastest lap (Driver A) ──────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(tel_a["Distance"], tel_a["Throttle"], color="green",  label="Throttle %", lw=1.2)
ax4.plot(tel_a["Distance"], tel_a["Brake"].astype(int) * 100,
         color="red", label="Brake (on/off)", lw=1.2, alpha=0.8)
ax4.set_xlabel("Distance (m)")
ax4.set_ylabel("Input (%)")
ax4.set_title(f"Throttle & Braking — {DRIVER_A} Fastest Lap")
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_ylim(-5, 110)

plt.savefig("../outputs/f1_overview.png", dpi=150, bbox_inches="tight")
print("\nChart saved: f1_overview.png")
plt.show()

# ── Summary stats printout ────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  QUICK STATS — {YEAR} {RACE} GP")
print(f"{'='*55}")
for drv in [DRIVER_A, DRIVER_B]:
    laps = session.laps.pick_drivers(drv).pick_quicklaps()
    best = laps["LapTime"].min()
    avg  = laps["LapTime"].mean()
    compounds = laps["Compound"].unique()
    print(f"  {drv}  |  Best: {best}  |  Avg: {avg}  |  Tires: {list(compounds)}")
print()
