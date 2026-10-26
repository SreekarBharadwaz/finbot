from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from nvidia_client import ask_nvidia, resolve_api_key
from chat_logic import (
    METRIC_LABELS,
    build_company_context,
    format_fact_value,
    latest_metric_fact,
    load_csv,
    sentiment_counts,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

st.set_page_config(page_title="FinSight — Financial Research Chat", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="Loading prepared SEC and transcript data…")
def get_data() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return (
        load_csv(DATA_DIR / "financial_facts.csv"),
        load_csv(DATA_DIR / "transcript_evidence.csv"),
    )


def api_setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(os.getenv(name, value) or default).strip()


def get_current_filing_period(facts: list[dict[str, str]], company: str) -> str:
    dates = [row.get("period_end", "") for row in facts if row.get("company", "").casefold() == company.casefold()]
    return max(dates) if dates else "Not available"


st.title("📊 FinSight")
st.caption("Chat with curated SEC filing facts and earnings-transcript findings. Financial data is reported as of each filing period shown.")
st.caption("General questions can be answered from model knowledge and will be labeled as outside your data. Missing company-specific or current figures will not be guessed.")

facts, evidence = get_data()
if not facts:
    st.error("Prepared SEC data is missing. Run `python prepare_data.py` from this project folder, then refresh the app.")
    st.stop()

companies = sorted({row["company"] for row in facts if row.get("company")})
with st.sidebar:
    st.header("Research scope")
    company = st.selectbox("Company", companies, index=companies.index("NVIDIA") if "NVIDIA" in companies else 0)
    peers = ["None"] + [item for item in companies if item != company]
    peer = st.selectbox("Compare with (optional)", peers, index=0)
    st.divider()
    session_api_key = st.text_input(
        "Your NVIDIA API key",
        type="password",
        help="Used only for your current browser session. Each user should enter their own key; it is not saved to the app's files.",
    )
    model = st.text_input("NVIDIA model ID", value=api_setting("NVIDIA_MODEL", DEFAULT_MODEL))
    st.caption("Get an API key from NVIDIA; requests are sent to NVIDIA's hosted inference service. For private deployments, you can alternatively configure NVIDIA_API_KEY in the host's Secrets settings.")
    st.caption("Model access and free trial quotas depend on your NVIDIA account and can change.")
    st.divider()
    st.caption("Source files are prepared from your local SEC and transcript-analysis exports; this app does not load the original 700+ MB SEC CSV at runtime.")

api_key = resolve_api_key(session_api_key, api_setting("NVIDIA_API_KEY"))
primary_period = get_current_filing_period(facts, company)
peer_period = get_current_filing_period(facts, peer) if peer != "None" else ""

st.subheader(f"{company} · SEC period {primary_period}")
metric_columns = st.columns(4)
key_metrics = ["revenue", "net_income", "assets", "operating_cash_flow"]
for column, metric in zip(metric_columns, key_metrics):
    row = latest_metric_fact(facts, company, metric)
    with column:
        label = METRIC_LABELS[metric]
        if row:
            period_quarters = str(row.get("period_quarters", ""))
            duration = "balance-sheet date" if period_quarters == "0" else (
                "quarter" if period_quarters == "1" else f"{period_quarters}-quarter reported period"
            )
            st.metric(label, format_fact_value(row), help=f"{duration}; period ending {row.get('period_end')}; accession {row.get('adsh')}")
        else:
            st.metric(label, "Not available")

with st.expander("View prepared financial facts and transcript coverage", expanded=False):
    st.markdown("**Financial facts**")
    selected_facts = [row for row in facts if row.get("company", "").casefold() == company.casefold()]
    st.dataframe(selected_facts, use_container_width=True, hide_index=True)
    st.markdown("**Transcript sentiment labels** (classification counts only; not share-price sentiment or a recommendation)")
    counts = sentiment_counts(evidence, company)
    if counts:
        st.json(counts)
    else:
        st.info("No normalized transcript sentiment labels are available for this company.")
    st.caption("Quarterly/YTD durations and balance-sheet values are deliberately kept separate. The notebook's old annual `aggfunc='first'` table and health scores are not used by this app.")

st.markdown("### Ask about the filings and transcript evidence")
if not api_key:
    st.info("Enter your NVIDIA API key in the password field in the sidebar to enable LLM answers, or configure NVIDIA_API_KEY in the deployment's Secrets settings.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about revenue, profitability, cash flow, risks, or management guidance…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if not api_key:
        answer = "No API key is set. Enter your NVIDIA key in the sidebar and submit your question again."
        with st.chat_message("assistant"):
            st.warning(answer)
    else:
        context = build_company_context(company, facts, evidence, prompt)
        if peer != "None":
            context += "\n\nPEER COMPANY CONTEXT\n" + build_company_context(peer, facts, evidence, prompt)
        prior_turns = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages[:-1]
            if item["role"] in {"user", "assistant"}
        ]
        with st.chat_message("assistant"):
            with st.spinner(f"Querying NVIDIA model {model}…"):
                try:
                    answer = ask_nvidia(prompt, context, prior_turns, api_key, model)
                    st.markdown(answer)
                except Exception as exc:
                    answer = "The NVIDIA model request failed. Check the model ID, NVIDIA API key, account quota, and network, then retry."
                    st.error(answer)
                    st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption("Educational research tool only—not investment advice. SEC facts and transcript labels are limited to the provided exports and may be incomplete or stale; verify original filings before relying on them.")
