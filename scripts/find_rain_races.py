"""Quick scan: which 2024 races had meaningful rainfall?"""
import fastf1, pandas as pd, warnings
warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache("../cache")

ALL_RACES = [
    "Bahrain","Saudi Arabia","Australia","Japan","China",
    "Miami","Monaco","Canada","Spain","Austria","Britain",
    "Hungary","Belgium","Netherlands","Italy","Azerbaijan",
    "Singapore","United States","Mexico","Brazil","Las Vegas","Qatar","Abu Dhabi"
]

print(f"\n{'Race':<18} {'MaxRain':>8} {'Rainfall laps':>14} {'TrackTemp min/max':>20}")
print("-"*65)
for race in ALL_RACES:
    try:
        s = fastf1.get_session(2024, race, "R")
        s.load(telemetry=False, messages=False)
        w = s.weather_data
        rain_max  = w["Rainfall"].max() if "Rainfall" in w.columns else 0
        rain_pct  = (w["Rainfall"] > 0).mean() * 100 if "Rainfall" in w.columns else 0
        tmin = w["TrackTemp"].min(); tmax = w["TrackTemp"].max()
        flag = " *** WET ***" if rain_pct > 5 else (" ~ damp" if rain_pct > 0 else "")
        print(f"{race:<18} {rain_max:>8.2f} {rain_pct:>13.1f}%  {tmin:>8.1f} / {tmax:<6.1f}{flag}")
    except Exception as e:
        print(f"{race:<18} ERROR: {e}")
