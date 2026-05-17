# F1 Telemetry Analysis

Data science project using the [FastF1](https://docs.fastf1.dev/) Python library to analyse Formula 1 race telemetry.

## What it does

**`scripts/f1_overview.py`** — Data exploration & visualisation
- Lap times across a full race
- Tyre degradation by stint
- Speed trace comparison between two drivers on their fastest lap
- Throttle and braking input trace

**`scripts/f1_prediction.py`** — Machine learning model
- Trains a Gradient Boosting model across multiple races
- Target: seconds above/below each circuit's median lap time (removes circuit-to-circuit baseline gap so the model learns degradation, not geography)
- Features: tyre life, compound, fuel load, track temp, air temp, driver, lap number, compound×tyre life interaction
- Outputs: feature importance, actual vs predicted, error distribution, per-compound degradation projection

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd scripts

# Telemetry overview (change YEAR, RACE, DRIVER_A/B at top of file)
python3 f1_overview.py

# Prediction model (change RACES list at top of file)
python3 f1_prediction.py
```

Charts save to `outputs/`. Race data is cached in `cache/` after first download.

## Model results (2024 — Bahrain, Saudi Arabia, Australia, Japan)

| Metric | Value |
|--------|-------|
| MAE | 0.38 sec |
| R² | 0.78 |

**Top predictors of lap time variation within a race:**

| Feature | Importance | Why |
|---------|-----------|-----|
| Fuel load | 22% | Car is ~110 kg heavier at race start; burns ~1.6 kg/lap |
| Lap number | 21% | Correlated with fuel, safety cars, track evolution |
| Air temp | 18% | Affects tyre operating window |
| Track temp | 15% | Surface grip changes through the race |
| Driver | 10% | Pace differences and tyre management style |
| Compound | 6% | SOFT vs HARD baseline delta |
| Tyre life | 3% | Degradation effect (smaller than fuel within a stint) |

## Key findings

- **Fuel burn dominates within-race lap time variation** — bigger effect than tyre degradation
- **Tyre compound matters more for strategy** than for raw within-stint lap time
- The out-lap (lap 1 of every stint) is genuinely slow for all compounds — cold rubber on cold tarmac
- SOFT tyres have limited real data beyond ~15 laps (teams rarely run them that long), so the model extrapolates noisily past that point

## Data source

FastF1 pulls official F1 timing and telemetry data. All data is publicly available and cached locally.
