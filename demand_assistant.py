"""Natural-language demand assistant for the operations manager.

Answers questions like "How many bikes do we need next Friday at 6pm if it rains?" from the
Part 3 model (models/bike_demand.joblib). Two modes share the same tool functions:

- offline (default): a rule-based parser maps the question to a tool call. No API key needed,
  fully deterministic, so every answer can be traced back to a model output.
- gemini: a LangGraph agent with Gemini (via LangChain) reads the question, decides which tools
  to call, and writes the answer. Used when langgraph + langchain-google-genai are installed
  and GOOGLE_API_KEY (or GEMINI_API_KEY) is set.

Every number in an answer comes from a tool, never from the language layer.
"""
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import bike_model as bm

HERE = Path(__file__).parent
MODEL = joblib.load(HERE / "models" / "bike_demand.joblib")
HOUR = pd.read_csv(HERE / "hour.csv", parse_dates=["dteday"])
HOUR["in_use"] = bm.bikes_in_use(HOUR.registered, HOUR.casual)
HOUR["temp_c"] = HOUR.temp * 41
HOUR["hum_pct"] = HOUR.hum.replace(0, np.nan) * 100
HOUR["wind_kmh"] = HOUR.windspeed.replace(0, np.nan) * 67

# Typical conditions per month, used when a question doesn't give the temperature
CLIMATE = HOUR[HOUR.hr.between(8, 20)].groupby("mnth")[["temp_c", "hum_pct", "wind_kmh"]].median()

WEATHER_CODES = {"clear": 1, "cloudy": 2, "rain": 3, "heavy rain": 4}
DESIGN_IN_USE = round(float(MODEL["design_in_use"]))
FLEET = float(MODEL["fleet"])
DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


@dataclass
class Answer:
    text: str
    table: pd.DataFrame | None = None
    calls: list = field(default_factory=list)  # (tool name, arguments) actually executed
    mode: str = "offline"


# ---- Tools: the only source of numbers ------------------------------------------------

def us_holiday(d):
    """Name of the US federal holiday on date d (the dataset's holiday calendar is D.C.'s), else None."""
    nth = lambda n, wd: (d.weekday() == wd and (d.day - 1) // 7 == n - 1)  # n-th given weekday of the month
    last_mon = d.weekday() == 0 and d.day + 7 > pd.Timestamp(d).days_in_month
    fixed = {(1, 1): "New Year's Day", (7, 4): "Independence Day", (11, 11): "Veterans Day", (12, 25): "Christmas Day"}
    floating = {(1, nth(3, 0)): "Martin Luther King Jr. Day", (2, nth(3, 0)): "Presidents' Day",
                (5, last_mon): "Memorial Day", (9, nth(1, 0)): "Labor Day",
                (10, nth(2, 0)): "Columbus Day", (11, nth(4, 3)): "Thanksgiving"}
    return fixed.get((d.month, d.day)) or floating.get((d.month, True))


def forecast_day(date_str, weather="clear", temp_c=None, holiday=False, growth_pct=0):
    """Hourly forecast for one day: rentals, bikes in use, and the 90th-percentile bikes to plan for."""
    d = pd.Timestamp(date_str)
    holiday_name = us_holiday(d.date())
    holiday = bool(holiday or holiday_name)
    code = WEATHER_CODES[weather]
    clim = CLIMATE.loc[d.month]
    temp = clim.temp_c if temp_c is None else float(temp_c)
    s = bm.scenario_hours(d, code, temp, clim.hum_pct, clim.wind_kmh, holiday)
    X = bm.make_features(s)
    g = 1 + growth_pct / 100
    reg = MODEL["registered"].predict(X) * g
    cas = MODEL["casual"].predict(X) * g
    plan = MODEL["in_use_q90"].predict(X) * MODEL["calibration_factor"] * g
    table = pd.DataFrame({
        "hour": range(24),
        "expected rentals": reg + cas,
        "members": reg,
        "casual": cas,
        "bikes in use (expected)": bm.bikes_in_use(reg, cas),
        "bikes to plan for (90%)": plan,
    }).round(0).astype(int)
    peak = int(table["bikes to plan for (90%)"].idxmax())
    return {
        "date": d.strftime("%A %d %B %Y"),
        "day_type": "working day" if s.workingday[0] else (holiday_name or "public holiday" if holiday else "weekend"),
        "weather": weather,
        "temp_c": round(float(temp), 1),
        "temp_assumed": temp_c is None,
        "growth_pct": growth_pct,
        "beyond_training_data": d.year > 2012,
        "total_rentals": int(table["expected rentals"].sum()),
        "peak_hour": peak,
        "peak_bikes_to_plan": int(table.loc[peak, "bikes to plan for (90%)"]),
        "peak_share_of_design_capacity": round(table.loc[peak, "bikes to plan for (90%)"] / DESIGN_IN_USE, 2),
        "hourly": table,
    }


def historical_demand(by="hr", metric="rentals", day_type=None, weather=None, season=None, year=None):
    """Average demand in 2011-2012, grouped by hour / weekday / month / season / weather / day type."""
    h = HOUR
    if day_type:
        h = h[h.workingday == (1 if day_type == "working day" else 0)]
    if weather:
        h = h[h.weathersit == WEATHER_CODES[weather]]
    if season:
        h = h[h.season == {v.lower(): k for k, v in bm.SEASON_NAMES.items()}[season]]
    if year:
        h = h[h.yr == int(year) - 2011]
    col = "in_use" if metric == "bikes in use" else "cnt"
    out = h.groupby(by)[col].mean().round(0).astype(int)
    labels = {"weekday": dict(enumerate(DAY_NAMES)), "mnth": {i + 1: m.title() for i, m in enumerate(MONTHS)},
              "season": bm.SEASON_NAMES, "weathersit": bm.WEATHER_NAMES,
              "workingday": {0: "Weekend/holiday", 1: "Working day"}}
    if by in labels:
        out.index = out.index.map(labels[by])
    names = {"hr": "hour", "weekday": "weekday", "mnth": "month", "season": "season",
             "weathersit": "weather", "workingday": "day type"}
    table = out.rename(f"avg {metric} per hour").rename_axis(names[by]).reset_index()
    return {"filters": {k: v for k, v in dict(day_type=day_type, weather=weather, season=season, year=year).items() if v},
            "metric": metric, "by": names[by], "hours_in_sample": int(len(h)),
            "overall_avg": int(round(h[col].mean())), "table": table}


def fleet_summary():
    """Fleet-size recommendation from Part 3, with utilisation and the coverage trade-off."""
    h12 = HOUR[HOUR.yr == 1]
    return {
        "fleet_needed": int(round(FLEET, -1)),
        "design_peak_bikes_on_road": DESIGN_IN_USE,
        "plausible_range": "600-1,100 depending on trip length and parked-bike buffer",
        "avg_bikes_in_use_2012": int(h12.in_use.mean()),
        "utilisation": round(h12.in_use.mean() / FLEET, 2),
        "pressure_windows": "working days 17:00-18:00 and weekend/holiday afternoons 11:00-17:00",
        "assumptions": {"trip_min_member": bm.TRIP_MIN["registered"], "trip_min_casual": bm.TRIP_MIN["casual"],
                        "parked_bikes_per_bike_in_use": bm.AVAILABILITY_BUFFER, "in_repair": bm.MAINTENANCE_SHARE},
        "tradeoff": MODEL["tradeoff"].round(0).astype(int).reset_index(),
    }


TOOLS = {"forecast_day": forecast_day, "historical_demand": historical_demand, "fleet_summary": fleet_summary}


# ---- Offline mode: rule-based question parser -----------------------------------------

def parse_date(q, today):
    if m := re.search(r"\b(20\d\d)-(\d\d)-(\d\d)\b", q):
        return date(*map(int, m.groups()))
    if "day after tomorrow" in q:
        return today + timedelta(days=2)
    if "tomorrow" in q:
        return today + timedelta(days=1)
    if "today" in q or "tonight" in q:
        return today
    month_re = "|".join(MONTHS + [m[:3] for m in MONTHS])
    m = (re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_re})\b(?:\s+(20\d\d))?", q)
         or re.search(rf"\b({month_re})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(20\d\d))?", q))
    if m:
        a, b, year = m.groups()
        day_s, mon_s = (a, b) if a.isdigit() else (b, a)
        month = next(i + 1 for i, name in enumerate(MONTHS) if name.startswith(mon_s[:3]))
        d = date(int(year) if year else today.year, month, int(day_s))
        return d if year or d >= today else d.replace(year=today.year + 1)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if m := re.search(r"\b(next|this|on|coming)?\s*(" + "|".join(days) + r")\b", q):
        target = days.index(m.group(2))
        ahead = (target - today.weekday()) % 7
        if ahead == 0 and m.group(1) == "next":
            ahead = 7
        return today + timedelta(days=ahead)
    if "weekend" in q and re.search(r"\b(this|next|coming)\s+weekend", q):
        return today + timedelta(days=(5 - today.weekday()) % 7 or 7)
    return None


def parse_weather(q):
    if re.search(r"heavy rain|storm|blizzard|thunder|downpour", q):
        return "heavy rain"
    if re.search(r"\brain|rainy|raining|snow|drizzle|shower|wet\b", q):
        return "rain"
    if re.search(r"cloud|mist|fog|overcast|grey|gray", q):
        return "cloudy"
    if re.search(r"sunny|clear|dry|nice weather|good weather", q):
        return "clear"
    return None


def parse_hour(q):
    if m := re.search(r"\b(\d{1,2})(?::(\d\d))?\s*(am|pm)\b", q):
        h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
        return h
    if m := re.search(r"\b([01]?\d|2[0-3]):\d\d\b", q):
        return int(m.group(1))
    if "morning peak" in q or "morning rush" in q:
        return 8
    if "evening peak" in q or "evening rush" in q or "after work" in q:
        return 17
    return None


def parse_temp(q):
    if m := re.search(r"(-?\d{1,2})\s*(?:°\s*c?|degrees?|deg|c\b)", q):
        return float(m.group(1))
    for word, t in [("freezing", 2), ("cold", 6), ("hot", 32), ("warm", 26), ("mild", 18)]:
        if re.search(rf"\b{word}\b", q):
            return t
    return None


def route(question, today):
    """Map a question to (tool name, arguments), or (None, reason)."""
    q = question.lower()
    d = parse_date(q, today)
    weather = parse_weather(q)
    growth = int(m.group(1)) if (m := re.search(r"(\d{1,3})\s*%\s*(?:more|growth|higher|increase)", q)) else 0
    if d:
        args = {"date_str": d.isoformat(), "weather": weather or "clear", "temp_c": parse_temp(q),
                "holiday": "holiday" in q, "growth_pct": growth}
        compare = weather and weather != "clear" and re.search(r"\bvs\b|versus|compare|difference|instead|how much (?:less|fewer|lower)", q)
        return "forecast_day", {**args, "_hour": parse_hour(q), "_compare": bool(compare)}
    if re.search(r"how many bikes|fleet|buy|purchase|more bikes|enough bikes|utili[sz]|idle|sitting unused|capacity", q) \
            and not re.search(r"average|typical|usually|histor", q):
        return "fleet_summary", {}
    by = ("weekday" if re.search(r"day of (?:the )?week|which day|weekday\b(?! vs)|by day", q)
          else "mnth" if re.search(r"month", q)
          else "season" if re.search(r"season", q) and not re.search(r"\b(?:in|during)\s+(?:winter|spring|summer|fall|autumn)", q)
          else "weathersit" if re.search(r"weather|rain affect|effect of rain|impact of rain", q)
          else "workingday" if re.search(r"weekends? (?:vs|versus|or|compared)|weekdays? (?:vs|versus|or|compared)|working days? vs", q)
          else "hr")
    hist_words = (r"average|typical|usually|normally|busiest|quietest|peak|when|which|compare|histor|pattern|rush|"
                  r"demand|trend|affect|effect|impact|rentals|usage|riders|20(11|12)")
    if re.search(hist_words, q) or weather:
        day_type = ("weekend/holiday" if re.search(r"weekend|holiday|saturday|sunday", q) and by != "workingday"
                    else "working day" if re.search(r"weekday|working day|workday|commut", q) and by != "workingday"
                    else None)
        season = next((s for s in ["winter", "spring", "summer", "fall"] if s in q or (s == "fall" and "autumn" in q)), None)
        if by == "season":
            season = None
        year = int(m.group(0)) if (m := re.search(r"\b20(11|12)\b", q)) else None
        metric = "bikes in use" if re.search(r"bikes? (?:in use|on the road|out)|bikes needed", q) else "rentals"
        return "historical_demand", {"by": by, "metric": metric, "day_type": day_type,
                                     "weather": weather if by != "weathersit" else None,
                                     "season": season, "year": year}
    return None, "unrecognised"


def _fmt_forecast(r, hour=None):
    lines = [f"**{r['date']}** ({r['day_type']}, {r['weather']}, {r['temp_c']}°C"
             + (" typical for the month" if r["temp_assumed"] else "") + ")"]
    if hour is not None:
        row = r["hourly"].loc[hour]
        lines.append(f"- At **{hour:02d}:00**: about **{row['expected rentals']}** rentals; plan for "
                     f"**{row['bikes to plan for (90%)']} bikes on the road** (expected {row['bikes in use (expected)']}).")
    lines.append(f"- Busiest hour: **{r['peak_hour']:02d}:00**, plan for **{r['peak_bikes_to_plan']} bikes on the road** "
                 f"({r['peak_share_of_design_capacity']:.0%} of the fleet's design capacity of {DESIGN_IN_USE:.0f}).")
    lines.append(f"- Whole day: about **{r['total_rentals']:,}** rentals.")
    if r["peak_share_of_design_capacity"] > 0.95:
        lines.append("- ⚠️ Near capacity: pre-position bikes before the peak at high-demand stations.")
    notes = ["90% plan = demand stays below this 9 hours out of 10"]
    if r["beyond_training_data"]:
        notes.append("model trained on 2011-2012, so this is at 2012 demand levels"
                     + (f" plus {r['growth_pct']}% growth" if r["growth_pct"] else "; add a growth % if demand has grown"))
    if r["weather"] == "heavy rain":
        notes.append("heavy rain occurred in only 3 training hours, so treat this as rough")
    lines.append("\n_Assumptions: " + "; ".join(notes) + "._")  # blank line keeps it out of the bullet list
    return "\n".join(lines)


def answer_offline(question, today=None):
    today = today or date.today()
    tool, args = route(question, today)
    if tool is None:
        return Answer("I can answer questions about **forecast demand** for a date (\"How many bikes do we need "
                      "next Friday at 6pm if it rains?\"), **historical patterns** (\"What are the busiest hours on "
                      "weekends?\") and **fleet size** (\"Do we need to buy more bikes?\"). Could you rephrase?")
    if tool == "forecast_day":
        hour, compare = args.pop("_hour"), args.pop("_compare")
        r = forecast_day(**args)
        calls = [(tool, args)]
        text = _fmt_forecast(r, hour)
        table = r["hourly"]
        if compare:
            base_args = {**args, "weather": "clear"}
            b = forecast_day(**base_args)
            calls.append((tool, base_args))
            diff = r["peak_bikes_to_plan"] - b["peak_bikes_to_plan"]
            text += (f"\n\n**Compared with clear weather:** peak plan {b['peak_bikes_to_plan']} → {r['peak_bikes_to_plan']} "
                     f"bikes ({diff:+d}); daily rentals {b['total_rentals']:,} → {r['total_rentals']:,} "
                     f"({r['total_rentals'] / b['total_rentals'] - 1:+.0%}).")
            table = table[["hour", "bikes to plan for (90%)"]].rename(columns={"bikes to plan for (90%)": f"plan: {args['weather']}"})
            table["plan: clear"] = b["hourly"]["bikes to plan for (90%)"]
        return Answer(text, table, calls)
    if tool == "fleet_summary":
        r = fleet_summary()
        text = (f"**Recommended fleet: about {r['fleet_needed']:,} bikes** ({r['plausible_range']}).\n"
                f"- That covers **{r['design_peak_bikes_on_road']} bikes on the road at once** on 95% of days, "
                f"plus parked bikes and repairs.\n"
                f"- In 2012 an average of only **{r['avg_bikes_in_use_2012']}** bikes were on trips: "
                f"each bike is used **~{r['utilisation']:.0%}** of the time.\n"
                f"- The fleet is only tested in **{r['pressure_windows']}**.\n"
                "- **Before buying:** if the current fleet is already above this, shortages are about *where* bikes are, "
                "so rebalance before the peaks rather than buy. The table shows the cost of covering more busy days.")
        return Answer(text, r["tradeoff"], [(tool, args)])
    r = historical_demand(**args)
    t = r["table"]
    col = t.columns[1]
    top = t.nlargest(3, col)
    low = t.nsmallest(1, col).iloc[0]
    label = lambda v: f"{v:02d}:00" if r["by"] == "hour" else str(v)
    filters = dict(r["filters"])
    period = str(filters.pop("year")) if "year" in filters else "2011–2012"
    filt = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in filters.items()) or "all days"
    head = f"**Average {r['metric']} per hour by {r['by']}** ({period}, {filt}; {r['hours_in_sample']:,} hours):\n"
    if len(t) <= 4:  # few groups: list them all, highest first
        body = "".join(f"- **{label(row.iloc[0])}**: {row.iloc[1]}\n" for _, row in t.sort_values(col, ascending=False).iterrows())
    else:
        body = ("- Highest: " + ", ".join(f"**{label(row.iloc[0])}** ({row.iloc[1]})" for _, row in top.iterrows()) + "\n"
                f"- Lowest: **{label(low.iloc[0])}** ({low.iloc[1]})\n")
    text = head + body + f"- Overall average: {r['overall_avg']} per hour."
    return Answer(text, t, [(tool, args)])


# ---- Gemini mode: a LangGraph agent chooses the tools and writes the answer -------------

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
SYSTEM_PROMPT = """You help the operations manager of a city bike-sharing company understand demand.

Use the tools for every number you give. Never estimate, extrapolate, or recall figures yourself; if no tool can answer, say what you can answer instead.

Context:
- The model was trained on Washington D.C. Capital Bikeshare data from 2011-2012. Forecasts for later dates are at 2012 demand levels unless the user gives a growth percentage; say so.
- "Bikes to plan for (90%)" means demand stays below this level 9 hours out of 10.
- Bikes in use assumes 12-minute member trips and 30-minute casual trips.
- The recommended fleet (~720 bikes, range 600-1,100) covers 95% of days; utilisation is low and pressure is concentrated in working-day evenings and weekend afternoons.
- When the user gives no temperature, the tool uses the typical temperature for that month.
- Heavy rain occurred in only 3 training hours, so flag such forecasts as rough.

Answer in a few short bullet points: the number first, then the one or two caveats that matter."""

MAX_GRAPH_STEPS = 12  # LangGraph recursion limit: bounds the agent <-> tools loop


def _to_json(result):
    """Tool result as JSON for the LLM; tables go in as a list of rows."""
    payload = {k: v for k, v in result.items() if not isinstance(v, pd.DataFrame)}
    table = next((v for v in result.values() if isinstance(v, pd.DataFrame)), None)
    if table is not None:
        payload["rows"] = table.to_dict(orient="records")
    return json.dumps(payload, default=str)


def gemini_available():
    """True when the LangGraph/Gemini packages are installed and an API key is set."""
    try:
        import langchain_google_genai  # noqa: F401
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def make_lc_tools():
    """The three tools wrapped as LangChain tools; the schemas come from the type hints and docstrings."""
    from typing import Literal, Optional
    from langchain_core.tools import tool

    Weather = Literal["clear", "cloudy", "rain", "heavy rain"]

    @tool("forecast_day", parse_docstring=True)
    def forecast_day_tool(date_str: str, weather: Weather = "clear", temp_c: Optional[float] = None,
                          holiday: bool = False, growth_pct: float = 0) -> str:
        """Forecast hourly bike demand for one calendar day: expected rentals, bikes in use, and the 90th-percentile bikes to plan for in each hour, plus the peak hour.

        Args:
            date_str: Date as YYYY-MM-DD. Resolve relative dates using today's date from the user message.
            weather: Weather for the day.
            temp_c: Air temperature in Celsius; leave empty for the month's typical value.
            holiday: True if the date is a public holiday (US federal holidays are detected automatically).
            growth_pct: Demand growth versus 2012 in percent; 0 if the user didn't give one.
        """
        return _to_json(forecast_day(date_str, weather, temp_c, holiday, growth_pct))

    @tool("historical_demand", parse_docstring=True)
    def historical_demand_tool(by: Literal["hr", "weekday", "mnth", "season", "weathersit", "workingday"] = "hr",
                               metric: Literal["rentals", "bikes in use"] = "rentals",
                               day_type: Optional[Literal["working day", "weekend/holiday"]] = None,
                               weather: Optional[Weather] = None,
                               season: Optional[Literal["winter", "spring", "summer", "fall"]] = None,
                               year: Optional[Literal[2011, 2012]] = None) -> str:
        """Average hourly demand in 2011-2012 grouped by one dimension, with optional filters. Use for typical patterns, busiest times, or the effect of weather or season.

        Args:
            by: Group by hour (hr), weekday, month (mnth), season, weather (weathersit), or working day vs weekend (workingday).
            metric: Rentals per hour, or bikes in use per hour.
            day_type: Only working days, or only weekends and holidays.
            weather: Only hours with this weather.
            season: Only this season.
            year: Only this year.
        """
        return _to_json(historical_demand(by, metric, day_type, weather, season, year))

    @tool("fleet_summary")
    def fleet_summary_tool() -> str:
        """The recommended fleet size, utilisation, pressure windows, sizing assumptions, and the trade-off between fleet size and the number of busy days covered."""
        return _to_json(fleet_summary())

    return [forecast_day_tool, historical_demand_tool, fleet_summary_tool]


def build_agent(llm=None):
    """LangGraph agent: an LLM node that may call tools, a tool node, and a loop between them.

        START -> agent --(tool calls?)--> tools -> agent ... --(no tool calls)--> END

    An in-memory checkpointer keeps each conversation (thread_id) so follow-ups work.
    `llm` can be any LangChain chat model with tool calling; the default is Gemini.
    """
    from langchain_core.messages import SystemMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    tools = make_lc_tools()
    if llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))  # errors go back to the LLM to self-correct
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # -> "tools" if the LLM asked for tools, else END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=InMemorySaver())


_AGENT = None


def answer_gemini(question, thread_id="default", today=None, agent=None):
    """Answer with the LangGraph agent. Turns with the same thread_id share conversation memory."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.errors import GraphRecursionError

    global _AGENT
    if agent is None:
        _AGENT = _AGENT or build_agent()
        agent = _AGENT
    today = today or date.today()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_GRAPH_STEPS}
    before = len(agent.get_state(config).values.get("messages", []))
    try:
        state = agent.invoke({"messages": [HumanMessage(f"Today is {today:%A %d %B %Y}.\n\n{question}")]}, config)
    except GraphRecursionError:
        return Answer("Stopped after too many tool rounds without a final answer.", None, [], "gemini")
    new = state["messages"][before:]  # only this turn's messages, not the stored history
    ok = {m.tool_call_id: m for m in new if isinstance(m, ToolMessage) and m.status != "error"}
    done = [tc for m in new if isinstance(m, AIMessage) for tc in m.tool_calls if tc["id"] in ok]
    calls = [(tc["name"], tc["args"]) for tc in done]
    tables = [(tc["args"], pd.DataFrame(rows)) for tc in done
              if (rows := json.loads(ok[tc["id"]].content).get("rows"))]
    plans = [(args, t) for args, t in tables if "bikes to plan for (90%)" in t.columns]
    if len(plans) > 1:  # a what-if comparison: one planning line per scenario
        table = pd.DataFrame({"hour": range(24)})
        for args, t in plans:
            label = " ".join(str(args[k]) for k in ("date_str", "weather", "temp_c") if args.get(k) is not None)
            table[f"plan: {label}"] = t["bikes to plan for (90%)"]
    else:
        table = tables[-1][1] if tables else None
    return Answer(new[-1].text, table, calls, "gemini")


def ask(question, mode="auto", thread_id="default", today=None):
    """Answer a question. mode: 'offline', 'gemini', or 'auto' (Gemini if available, else offline)."""
    if mode == "gemini" or (mode == "auto" and gemini_available()):
        try:
            return answer_gemini(question, thread_id, today)
        except Exception as e:  # prototype: any LLM/API failure falls back to the deterministic parser
            if mode == "gemini":
                raise
            a = answer_offline(question, today)
            a.text += f"\n\n_(Gemini unavailable: {type(e).__name__}; answered offline.)_"
            return a
    return answer_offline(question, today)
