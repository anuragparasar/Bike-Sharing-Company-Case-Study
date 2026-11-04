# Bike-Sharing Company Case Study

> **"Riders keep complaining they can't find a bike when they need one. Meanwhile I see bikes sitting unused all over the city. Should we just buy more bikes?"**
> (the CEO)

An end-to-end analysis of two years of Capital Bikeshare (Washington D.C.) data:
- data-quality checks and exploration
- a statistically tested hypothesis and a field experiment design
- a demand model that estimates **how many bikes are needed**
- an AI assistant the operations manager can ask about demand in plain English

📄 **Full report:** [`Bike_Sharing_Case_Study_Report.pdf`](Case%20Study/Bike_Sharing_Case_Study_Report.pdf)

---

## The answer in one minute

**Not before checking two things.** Both complaints are true at once because demand is concentrated in a few hours.

| Finding | Number |
|---|---|
| Time an average bike is actually on a trip | **~8%** |
| Hours in which the fleet is near its limit | **~7%**: working-day evenings (17:00–18:00) and weekend afternoons |
| Share of a working day's rentals in its 3 busiest hours | **33.4%** vs 27.4% on weekends (p ≈ 10⁻⁷²) |
| Forecast accuracy on 4 unseen months | **R² 0.91** (baseline 0.68); peak-hour error 84 vs 131 rentals/hour |
| Fleet needed to cover 95% of days | **~720 bikes** (plausible range 600–1,100) |
| Extra bikes to cover 95% of days instead of 80% | only **~50** |

**Recommendation:**
1. Compare the real fleet with ~720. If it's already larger, shortages come from *where* bikes are, not how many, so **rebalance before the peaks instead of buying**.
2. Use trip time limits or pricing on weekend afternoons, when long leisure trips tie bikes up.
3. Run a rebalancing pilot (pre-peak rebalancing vs +10% bikes) before any large purchase.
4. Collect station-level data.

---

## Repository structure

```
├── Case Study/              The case brief and the final PDF report
├── Dataset/                 UCI Bike Sharing data: hour.csv, day.csv, Readme.txt
├── Model/                   bike_demand.joblib: trained models + design numbers (made by 3_model.ipynb)
├── 1_note.ipynb             Part 1: dataset, quality checks, feature preparation, exploration
├── 2_hypothesis.ipynb       Part 2: hypothesis, statistical tests, experiment design
├── 3_model.ipynb            Part 3: demand model, bikes-in-use, calibrated planning margin, fleet sizing
├── 4_prototype.ipynb        Part 4: assistant design, demo, parser evaluation, LangGraph agent test
├── bike_model.py            Shared features and fleet-sizing assumptions
├── demand_assistant.py      Tools, offline question parser, LangGraph + Gemini agent
└── app.py                   Streamlit chat app for the operations manager
```

---

## Part 1: The data and its quality ([`1_note.ipynb`](1_note.ipynb))

**Data:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset): 731 days / 17,379 hours of rentals (2011–2012), split into **registered members** and **casual riders**, with weather and calendar data.

**Quality findings:**
- No nulls or duplicates; `casual + registered = cnt` on every row.
- **165 hours missing:** mostly 02:00–05:00 (no rentals), plus outages such as Hurricane Sandy (29–30 Oct 2012).
- `windspeed = 0` in 2,180 hours and `hum = 0` in 22 hours: sensor gaps, treated as missing.
- **The dataset readme mislabels seasons.** Code 1 covers 21 Dec – 20 Mar, i.e. winter, not spring. Corrected labels: 1 = winter, 2 = spring, 3 = summer, 4 = fall.

**What the data can't show:**
- It has city-wide totals only: no stations, locations or fleet size.
- It records only rentals that happened, so true peak demand is understated.

**Exploration:**
- Rentals grew **+65%** from 2011 to 2012.
- Working days have sharp commute peaks at **08:00 and 17:00–18:00**; weekends have a broad midday hump.
- Rain roughly halves demand.
- On a typical day the busiest hour is **2.8×** the average hour.

## Part 2: A testable hypothesis ([`2_hypothesis.ipynb`](2_hypothesis.ipynb))

> **H₀:** the share of a day's rentals in its 3 busiest hours is the same on working days and non-working days.
> **H₁:** it is higher on working days, i.e. demand is more concentrated in time.

| Test | Result |
|---|---|
| Median top-3-hour share | 33.4% vs 27.4% (+6.0 points, bootstrap 95% CI +5.2 to +6.7) |
| Mann–Whitney U (one-sided) | p ≈ 2×10⁻⁷², rank-biserial effect size 0.83 |
| Permutation test | 0 of 10,000 shuffles reached the observed gap |
| Regression controlling for volume, weather, season, year | working-day peak hour **+28.5%** |
| Robustness | holds in all 8 season × year groups |

**H₀ is rejected.**

**Field experiment design** to test "buy bikes or manage the peaks":
- **Design:** cluster-randomised trial across station zones, with three arms: control, pre-peak rebalancing, +10% bikes.
- **Primary metric:** empty-station minutes at peak.
- **Analysis:** difference-in-differences.
- **Size:** ~64 zones per arm for 80% power, with a switchback design as the fallback if zones are few.

## Part 3: How many bikes are needed? ([`3_model.ipynb`](3_model.ipynb))

1. **Forecast hourly rentals.** Gradient-boosted trees (`HistGradientBoostingRegressor`, Poisson loss), with separate models for members and casual riders. Inputs are calendar and weather only, so the model can answer what-if questions.
2. **Convert to bikes in use** (Little's law): rentals × trip length, assuming 12 min for members and 30 min for casual riders.
3. **Add a safety margin.** A 90th-percentile model of bikes in use, calibrated on held-out data so demand stays under the plan 9 hours out of 10. Test coverage after calibration: **91%**, against a 90% target.
4. **Size the fleet.** Cover the busiest hour on 95% of days, then add 1.5 parked bikes per bike in use and 10% in repair: **~720 bikes**.

**Validation:** time-based split, training on Jan 2011 – Aug 2012 and testing on Sep – Dec 2012.

| Model | MAE (rentals/hour) | RMSLE | R² | Peak-hour MAE |
|---|---|---|---|---|
| Baseline (recent average by hour × day type × weather) | 77.3 | 0.638 | 0.680 | 130.8 |
| **Gradient boosting, members + casual** | **42.3** | **0.377** | **0.911** | **83.9** |

The trip lengths and parked-bike buffer are assumptions, so a sensitivity table shows the fleet ranging from **~600 to ~1,100**.

## Part 4: Ask about demand in plain English ([`4_prototype.ipynb`](4_prototype.ipynb), [`app.py`](app.py))

A Streamlit chat app for the operations manager:

| Question | Tool |
|---|---|
| "How many bikes do we need next Friday at 6pm if it rains?" | `forecast_day` |
| "How much does rain change demand on 14 June compared to clear weather?" | `forecast_day` × 2 |
| "What are the busiest hours on weekends?" | `historical_demand` |
| "Do we need to buy more bikes?" | `fleet_summary` |

**Design principles:**
- **Every number comes from a tool, never from the language model.**
- Each answer shows an hourly chart against fleet capacity, the data behind it, and the exact tool calls.
- **Two language layers:**
  - **Offline** (rule-based, no API key): parses dates, weather, times, temperatures and growth. It got 19/19 standard phrasings right and 0/5 harder ones.
  - **Gemini via a LangGraph agent:** an agent node and a tool node loop until the answer is written. A checkpointer gives conversation memory, so follow-ups work. Tool errors go back to the model so it can correct itself, and the loop is capped at 12 steps. If Gemini is unavailable, the app falls back to offline mode.

---

## Running it

**Requirements:** Python 3.10+

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn joblib streamlit
```

For the Gemini agent (optional):

```bash
pip install langgraph langchain-google-genai
```

**Notebooks:** run in order, `1_note` → `2_hypothesis` → `3_model` → `4_prototype`. `3_model.ipynb` creates the saved model file that the prototype uses.

**Chat app:**

```bash
streamlit run app.py
```

To use Gemini instead of the offline parser, set an API key from [Google AI Studio](https://aistudio.google.com/), then choose **Gemini (LangGraph agent)** in the sidebar:

```bash
export GOOGLE_API_KEY=your-key        # Windows: set GOOGLE_API_KEY=your-key
```

Optionally pick a different model:

```bash
export GEMINI_MODEL=gemini-3.5-flash
```

---

## Assumptions and limitations

- City-wide data can't show **where** bikes run out, the most likely cause of "can't find a bike". Station-level data is the key next step.
- Recorded rentals understate true peak demand (failed rentals aren't logged), so the fleet estimate is a **floor**.
- Trip lengths (12 / 30 min) and the parked-bike buffer (1.5) are assumptions; the sensitivity table shows their effect.
- The model has seen only 2011–2012. Later dates are forecast at 2012 levels unless a growth % is given.

---

## Data source

Fanaee-T, Hadi, and Gama, Joao, *"Event labeling combining ensemble detectors and background knowledge"*, Progress in Artificial Intelligence (2013), pp. 1–15, Springer. [doi:10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3)

**Author:** [Anurag Parasar Mund](https://github.com/anuragparasar)
