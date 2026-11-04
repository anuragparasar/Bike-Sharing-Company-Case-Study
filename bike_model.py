"""Shared feature preparation and fleet-sizing assumptions for the bike demand model.

Used by 3_model.ipynb (training) and by the natural-language prototype (scenarios),
so both build features exactly the same way.
"""
import numpy as np
import pandas as pd

# ---- Fleet-sizing assumptions (agree these with operations) -------------------------
# The dataset has no trip durations, so these are assumptions, tested in the sensitivity table.
TRIP_MIN = {"registered": 12, "casual": 30}  # average trip length in minutes; members ride short commutes
AVAILABILITY_BUFFER = 1.5  # parked bikes needed per bike in use, so riders find one nearby
MAINTENANCE_SHARE = 0.10  # share of the fleet in repair at any time

FEATURES = ["hr", "workingday", "weekday", "holiday", "season", "mnth", "yr",
            "weathersit", "temp_c", "hum_pct", "wind_kmh"]

SEASON_NAMES = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}  # corrected labels, see note.ipynb 3d
WEATHER_NAMES = {1: "Clear", 2: "Mist/Cloudy", 3: "Light rain/snow", 4: "Heavy rain/snow"}


def make_features(df):
    """Model features from rows shaped like hour.csv. Zero humidity/wind are sensor gaps -> NaN."""
    return pd.DataFrame({
        "hr": df["hr"],
        "workingday": df["workingday"],
        "weekday": df["weekday"],
        "holiday": df["holiday"],
        "season": df["season"],
        "mnth": df["mnth"],
        "yr": df["yr"],
        "weathersit": df["weathersit"],
        "temp_c": df["temp"] * 41,
        "hum_pct": df["hum"].replace(0, np.nan) * 100,
        "wind_kmh": df["windspeed"].replace(0, np.nan) * 67,
    }, index=df.index)


def bikes_in_use(registered, casual):
    """Average bikes out on trips during an hour (Little's law: rentals per hour x trip length)."""
    return (registered * TRIP_MIN["registered"] + casual * TRIP_MIN["casual"]) / 60


def fleet_needed(peak_in_use, buffer=AVAILABILITY_BUFFER, maintenance=MAINTENANCE_SHARE):
    """Total fleet so that the peak in-use count is covered, with parked bikes and repairs on top."""
    return peak_in_use * (1 + buffer) / (1 - maintenance)


def season_of(date):
    """Season code as the dataset actually uses it (astronomical seasons, 1 = winter)."""
    md = (date.month, date.day)
    if md >= (12, 21) or md <= (3, 20):
        return 1
    if md <= (6, 20):
        return 2
    if md <= (9, 22):
        return 3
    return 4


def scenario_hours(date, weathersit=1, temp_c=20.0, hum_pct=60.0, wind_kmh=13.0, holiday=False):
    """24 rows shaped like hour.csv for one day under the given conditions.

    The model only saw 2011-2012, so later dates are treated as 2012 (yr = 1);
    growth beyond that is applied separately as a multiplier.
    """
    date = pd.Timestamp(date)
    weekday = (date.dayofweek + 1) % 7  # dataset convention: 0 = Sunday
    return pd.DataFrame({
        "hr": range(24),
        "workingday": int(weekday not in (0, 6) and not holiday),
        "weekday": weekday,
        "holiday": int(holiday),
        "season": season_of(date),
        "mnth": date.month,
        "yr": int(min(max(date.year - 2011, 0), 1)),
        "weathersit": weathersit,
        "temp": temp_c / 41,
        "hum": hum_pct / 100,
        "windspeed": wind_kmh / 67,
    })
