"""Chat prototype for the operations manager. Run with:  streamlit run app.py"""
import uuid

import streamlit as st

import demand_assistant as da

st.set_page_config(page_title="Bike Demand Assistant", page_icon="🚲", layout="wide")

EXAMPLES = [
    "How many bikes do we need next Friday at 6pm if it rains?",
    "Will we run out of bikes tomorrow morning peak?",
    "How much does rain change demand on 14 June compared to clear weather?",
    "Forecast for Saturday at 2pm, 30 degrees and sunny with 20% growth",
    "What are the busiest hours on weekends?",
    "How does weather affect rentals?",
    "Do we need to buy more bikes?",
]

with st.sidebar:
    st.header("🚲 Bike Demand Assistant")
    st.caption("Answers come from the Part 3 demand model. Every number is computed by a tool, never written freehand.")
    mode = st.radio("Language layer", ["Offline (rule-based)", "Gemini (LangGraph agent)"], index=0,
                    help="Offline needs no API key. Gemini mode runs a LangGraph agent with Gemini via LangChain; "
                         "it needs GOOGLE_API_KEY (or GEMINI_API_KEY). If it's unavailable the app falls back to offline.")
    if mode.startswith("Gemini") and not da.gemini_available():
        st.warning("Set GOOGLE_API_KEY (and `pip install langgraph langchain-google-genai`), then restart. Using offline.")
    elif mode.startswith("Gemini"):
        st.caption(f"Model: `{da.GEMINI_MODEL}` (override with the GEMINI_MODEL environment variable)")
    st.metric("Recommended fleet", f"{round(da.FLEET, -1):,.0f} bikes")
    st.metric("Design peak on the road", f"{da.DESIGN_IN_USE:,} bikes")
    st.subheader("Try asking")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    if st.button("Clear chat"):
        st.session_state.chat = []
        st.session_state.thread_id = str(uuid.uuid4())  # new LangGraph thread = fresh agent memory

st.session_state.setdefault("chat", [])
st.session_state.setdefault("thread_id", str(uuid.uuid4()))


def show(answer):
    st.markdown(answer.text)
    t = answer.table
    if t is not None and "hour" in t.columns and len(t) == 24:
        cols = [c for c in t.columns if c.startswith(("bikes to plan", "plan:"))]
        chart = t.set_index("hour")[cols].copy()
        chart["design capacity"] = da.DESIGN_IN_USE
        st.line_chart(chart, x_label="hour of day", y_label="bikes on the road")
    if t is not None:
        with st.expander("Data behind this answer"):
            st.dataframe(t, hide_index=True, use_container_width=True)
    if answer.calls:
        with st.expander(f"How this was computed ({answer.mode} mode)"):
            for name, args in answer.calls:
                st.code(f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})", language="python")


for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        show(turn["answer"]) if turn["role"] == "assistant" else st.markdown(turn["text"])

question = st.chat_input("Ask about demand, a date, the weather, or fleet size…") or st.session_state.pop("pending", None)
if question:
    st.session_state.chat.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Working it out…"):
            # the LangGraph checkpointer keeps this chat's history under its thread_id, so follow-ups work
            answer = da.ask(question, mode="auto" if mode.startswith("Gemini") else "offline",
                            thread_id=st.session_state.thread_id)
        show(answer)
    st.session_state.chat.append({"role": "assistant", "answer": answer})
