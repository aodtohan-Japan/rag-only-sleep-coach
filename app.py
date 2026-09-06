import gzip
import io
import os
import pickle
import re
import zlib
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION & SESSION STATE
# ==============================================================================
st.set_page_config(page_title="RAG-Only Sleep Coach", page_icon="🌙", layout="wide")

# Initialize dark mode state
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Layout Title and Dark Mode Toggle Button side-by-side
col_title, col_toggle = st.columns([0.8, 0.2])

with col_title:
    st.title("🌙 AI Sleep Coach (RAG-Only)")
    st.caption(
        "A Knowledge-Grounded Conversational AI Powered by Evidence-Based Sleep Guidelines"
    )

with col_toggle:
    st.write("##")  # Visual spacing alignment
    toggle_label = (
        "☀️ Light Mode" if st.session_state["dark_mode"] else "🌙 Dark Mode"
    )
    if st.button(toggle_label, key="mode_toggle_btn"):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()

# Dynamic Theme Variables based on Toggle State
if st.session_state["dark_mode"]:
    bg_color = "#0f172a"
    text_color = "#f8fafc"
    card_title_color = "#f8fafc"
    card_bg_blue = "#1e293b"
    card_bg_green = "#1e293b"
    card_bg_purple = "#1e293b"
    card_bg_slate = "#1e293b"
    card_border = "#3b82f6"
    textarea_bg = "#1e293b"
    textarea_text = "#f8fafc"
    textarea_border = "#475569"
    select_label_color = "#ffffff"
else:
    bg_color = "#ffffff"
    text_color = "#0f172a"
    card_title_color = "#0f172a"
    card_bg_blue = "#e0f2fe"
    card_bg_green = "#dcfce7"
    card_bg_purple = "#f3e8ff"
    card_bg_slate = "#f1f5f9"
    card_border = "#38bdf8"
    textarea_bg = "#ffffff"
    textarea_text = "#0f172a"
    textarea_border = "#94a3b8"
    select_label_color = "#0f172a"

# Custom CSS Injecting Dynamic Variables
st.markdown(
    f"""
<style>
    /* Global App Background and Text Overrides */
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    
    /* Radio Header and Label Text Styling */
    div[data-testid="stRadio"] > label {{
        font-size: 22px !important;
        font-weight: 800 !important;
        color: {text_color} !important;
        margin-bottom: 8px !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] label p {{
        font-size: 18px !important;
        font-weight: 600 !important;
        color: {text_color} !important;
    }}

    /* Selectbox Field Labels (Period, Hour, Minute) Styling */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p {{
        font-size: 16px !important;
        font-weight: 700 !important;
        color: {select_label_color} !important;
    }}

    /* Question Title Styling for HTML wrappers */
    .card-title {{
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0px;
        color: {card_title_color} !important;
    }}

    /* Target Streamlit Slider Value and Min/Max Endpoint Numbers */
    div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
    div[data-testid="stSlider"] div[data-testid="stTickBarMax"],
    div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stSlider"] span {{
        font-size: 18px !important;
        font-weight: 700 !important;
        color: {text_color} !important;
    }}

    /* Enlarged floating current value label above the slider thumb */
    div[data-testid="stSlider"] div[role="slider"] {{
        font-size: 18px !important;
        font-weight: 800 !important;
    }}

    /* Text Area Input Styling with Visible Lined Border */
    div[data-testid="stTextArea"] textarea {{
        font-size: 16px !important;
        background-color: {textarea_bg} !important;
        color: {textarea_text} !important;
        border: 2px solid {textarea_border} !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stTextArea"] textarea:focus {{
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }}

    /* Custom Prominent Dark Blue Button Styling */
    div.stButton > button {{
        background-color: #1e40af !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 18px !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        border: none !important;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.2s ease-in-out !important;
    }}

    /* Hover effect for submit buttons */
    div.stButton > button:hover {{
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        box-shadow: 0px 6px 14px rgba(0, 0, 0, 0.25) !important;
        transform: translateY(-1px);
    }}
</style>
""",
    unsafe_allow_html=True,
)

# Retrieve API Key securely from Streamlit Secrets
openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY", None)

if not openrouter_api_key:
    st.sidebar.warning("⚠️ OpenRouter API Key missing in Streamlit Secrets.")
else:
    st.sidebar.success("🔒 API Key loaded securely from Secrets!")

# ==============================================================================
# LOAD RAG KNOWLEDGE BASE (Auto-downloads raw binary from GitHub Releases)
# ==============================================================================
RELEASE_DOWNLOAD_URL = "https://github.com/aodtohan-Japan/rag-only-sleep-coach/releases/download/v1.0/lightweight_rag_components.pkl"
PICKLE_MAGIC_BYTES = (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05")


def decompress_if_needed(data: bytes) -> bytes:
    """Decompresses zlib/gzip data if magic pickle bytes are missing."""
    if data.startswith(PICKLE_MAGIC_BYTES):
        return data

    # Check for zlib header (0x78)
    if data.startswith(b"\x78"):
        try:
            return zlib.decompress(data)
        except Exception:
            pass

    # Check for gzip header
    try:
        return gzip.decompress(data)
    except Exception:
        pass

    return data


@st.cache_resource
def load_rag_artifact():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "lightweight_rag_components.pkl")

    if not os.path.exists(file_path):
        with st.spinner("Downloading RAG Knowledge Base from GitHub Release..."):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept-Encoding": "identity",  # Request uncompressed transfer
            }
            response = requests.get(
                RELEASE_DOWNLOAD_URL,
                headers=headers,
                allow_redirects=True,
                stream=True,
                timeout=30,
            )
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    with open(file_path, "rb") as f:
        payload = f.read()

    # Attempt decompression if needed
    payload = decompress_if_needed(payload)

    # Validate pickle header
    if not payload.startswith(PICKLE_MAGIC_BYTES):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise ValueError(
            f"Downloaded payload is not a valid pickle file (received header: {payload[:20]!r}). "
            "Please ensure the repository/release asset is public."
        )

    return pickle.loads(payload)


try:
    rag_payload = load_rag_artifact()

    if isinstance(rag_payload, dict):
        rag_chunks = rag_payload.get("chunks", rag_payload.get("documents", []))
    else:
        rag_chunks = rag_payload

    st.sidebar.success("✅ RAG Knowledge Base Loaded!")
except Exception as e:
    st.sidebar.error("❌ Knowledge Base Error")
    st.error(
        f"**Error loading RAG file (`{e}`):** Please verify the `RELEASE_DOWNLOAD_URL` link in `app.py`."
    )
    st.stop()


# ==============================================================================
# LIGHTWEIGHT KEYWORD MATCHING ENGINE
# ==============================================================================
def search_raw_text_chunks(query, chunks, top_k=3):
    stopwords = {
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "to",
        "then",
    }

    query_tokens = [
        word
        for word in re.findall(r"\b\w+\b", query.lower())
        if word not in stopwords and len(word) > 2
    ]

    if not query_tokens:
        query_tokens = [w for w in query.lower().split() if len(w) > 2]

    scored_chunks = []

    for item in chunks:
        text_content = item["text"] if isinstance(item, dict) else str(item)
        source_doc = (
            item.get("source", "Sleep Guideline")
            if isinstance(item, dict)
            else "Knowledge Base"
        )

        chunk_tokens = re.findall(r"\b\w+\b", text_content.lower())
        chunk_token_counts = Counter(chunk_tokens)

        overlap_score = sum(
            chunk_token_counts[token]
            for token in query_tokens
            if token in chunk_token_counts
        )
        scored_chunks.append((overlap_score, text_content, source_doc))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def render_time_picker(
    label_prefix, default_hour=10, default_minute=0, default_period="PM"
):
    col_period, col_hr, col_min = st.columns(3)

    with col_period:
        period = st.selectbox(
            "Period",
            ["AM", "PM"],
            index=0 if default_period == "AM" else 1,
            key=f"{label_prefix}_period",
        )
    with col_hr:
        hour_12 = st.selectbox(
            "Hour",
            list(range(0, 13)),
            index=default_hour,
            key=f"{label_prefix}_hour",
        )
    with col_min:
        minute = st.selectbox(
            "Minute",
            [f"{m:02d}" for m in range(60)],
            index=default_minute,
            key=f"{label_prefix}_minute",
        )

    hr_24 = hour_12 % 12
    if period == "PM":
        hr_24 += 12

    return hr_24, int(minute), f"{hour_12:02d}:{minute} {period}"


def clean_and_trim_response(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    if "Here's a thinking process:" in cleaned:
        parts = cleaned.split("Here's a thinking process:", 1)
        lines = parts[1].split("\n")
        final_lines = [
            line
            for line in lines
            if not re.match(r"^\s*(\d+\.|\*|\-|o)\s+", line)
        ]
        cleaned = " ".join(final_lines).strip()

    cleaned = re.sub(r"^\s*[\*\-\•\d\.]+\s*", "", cleaned).strip()

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(sentences) > 3:
        return " ".join(sentences[:3])
    return cleaned if cleaned else text


# ==============================================================================
# INTERFACE MAIN BODY
# ==============================================================================
st.header("AI Sleep Coach")
st.write(
    "Grounded in medical knowledge extracted from CDC, NSF, Harvard, and NIH guidelines."
)

mode = st.radio(
    "Select Coaching Strategy Mode:",
    [
        "Mode 1: Morning Check-in & Habit Reflection",
        "Mode 2: Bedtime Procrastination & Negotiation Coach",
    ],
)

if "Mode 1" in mode:
    st.info(
        "💡 **Goal:** Reflect on your previous night's sleep metrics and get personalized feedback to optimize your daytime energy and sleep habits."
    )
else:
    st.info(
        "💡 **Goal:** Resolve late-night bedtime procrastination by evaluating the cognitive trade-offs of delaying sleep tonight."
    )

st.markdown("---")

if "Mode 1" in mode:
    # Question Block 1
    st.markdown(
        f"""
    <div style="background-color: {card_bg_blue}; border: 2px solid {card_border}; border-radius: 28px; padding: 24px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
        <div class="card-title">Previous Night Bedtime</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    bed_hr, bed_min, bedtime_display = render_time_picker(
        "Previous Night Bedtime",
        default_hour=10,
        default_minute=0,
        default_period="PM",
    )

    # Question Block 2
    st.markdown(
        f"""
    <div style="background-color: {card_bg_green}; border: 2px solid {card_border}; border-radius: 28px; padding: 24px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
        <div class="card-title">Morning Wake Up Time</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    wake_hr, wake_min, wake_display = render_time_picker(
        "Morning Wake Up Time",
        default_hour=7,
        default_minute=0,
        default_period="AM",
    )

    # Question Block 3: Subjective Alertness Self-Report
    st.markdown(
        f"""
    <div style="background-color: {card_bg_purple}; border: 2px solid {card_border}; border-radius: 28px; padding: 24px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
        <div class="card-title">Rate your current alertness-sleepiness levels (1 = extremely alert; 9 = extremely sleepy)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    user_self_alertness = st.slider(
        "Rate your current alertness-sleepiness levels (1 = extremely alert; 9 = extremely sleepy)",
        min_value=1,
        max_value=9,
        value=5,
        step=1,
        label_visibility="collapsed",
    )

    # Question Block 4
    st.markdown(
        f"""
    <div style="background-color: {card_bg_slate}; border: 2px solid {card_border}; border-radius: 28px; padding: 24px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
        <div class="card-title">(REQUIRED) Type in your Sleep Question or Check-in Reflection</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    user_query = st.text_area(
        "(REQUIRED) Type in your Sleep Question or Check-in Reflection",
        placeholder="Type here...",
        height=120,
        label_visibility="collapsed",
    )

    if st.button(
        "SUBMIT RESPONSE to Generate Personalized Feedback", key="submit_mode_1"
    ):
        if not user_query.strip():
            st.error(
                "⚠️ **Input Required:** Please type a question or reflection in the box above before submitting."
            )
        else:
            t_bed = datetime(2026, 1, 1, bed_hr, bed_min)
            t_wake = datetime(2026, 1, 1, wake_hr, wake_min)
            if t_wake <= t_bed:
                t_wake += timedelta(days=1)

            sleep_duration = (t_wake - t_bed).total_seconds() / 3600.0

            st.info(
                f"⏱️ **Logged Sleep Duration:** **{sleep_duration:.1f} hrs** ({bedtime_display} to {wake_display}) | "
                f"**Self-Reported Alertness:** **{user_self_alertness} / 9**"
            )

            if not openrouter_api_key:
                st.error(
                    "API Key not found. Please set `OPENROUTER_API_KEY` in Streamlit secrets."
                )
            else:
                with st.spinner(
                    "Executing retrieval & querying OpenRouter..."
                ):
                    top_matches = search_raw_text_chunks(
                        user_query, rag_chunks, top_k=3
                    )
                    context_str = "\n\n".join(
                        [f"Source ({m[2]}): {m[1]}" for m in top_matches]
                    )

                    system_prompt = f"""You are an expert, empathetic sleep coach assistant.
CRITICAL OUTPUT CONSTRAINTS:
1. Output MUST be between 1 and 3 sentences total. 
2. Output ONLY the final advice aimed at the user.
3. DO NOT include reasoning, chain-of-thought, meta-commentary, introductory text, or closing remarks. 
4. Never show inner logic or reference these system constraints in the output.

USER METRICS:
- Total Sleep Duration: {sleep_duration:.1f} hours (Bedtime: {bedtime_display}, Wake time: {wake_display})
- Self-Reported Alertness/Sleepiness Level: {user_self_alertness}/9 (1 = Extremely Alert, 9 = Extremely Sleepy)

Directly acknowledge their logged sleep duration and self-reported alertness score. Provide evidence-based advice tailored to their subjective feeling and reflection using the scientific context below.
Write a supportive answer in maximum 3 sentences.

CONTEXT:
{context_str}

USER REFLECTION:
{user_query}"""

                    try:
                        url = "https://openrouter.ai/api/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {openrouter_api_key}",
                            "Content-Type": "application/json",
                        }
                        payload = {
                            "model": "nvidia/nemotron-3.5-lightning:free",
                            "messages": [
                                {"role": "user", "content": system_prompt}
                            ],
                            "max_tokens": 1000,
                        }

                        response = requests.post(
                            url, headers=headers, json=payload, timeout=12
                        )
                        response.raise_for_status()
                        res_json = response.json()
                        raw_ai_response = res_json["choices"][0]["message"][
                            "content"
                        ]

                        final_response = clean_and_trim_response(
                            raw_ai_response
                        )

                        st.success("### AI Coach Guidance")
                        st.write(final_response)

                        with st.expander("🔍 View Retrieved Knowledge Context"):
                            seen_sources = set()
                            for match in top_matches:
                                source_name = match[2]
                                if source_name not in seen_sources:
                                    st.markdown(f"• **{source_name}**")
                                    seen_sources.add(source_name)
                    except Exception as e:
                        st.error(f"OpenRouter API Error: {e}")

else:
    # Question Block 1
    st.markdown(
        f"""
    <div style="background-color: {card_bg_green}; border: 2px solid {card_border}; border-radius: 28px; padding: 24px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
        <div class="card-title">What time is it now?</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    now_hr, now_min, now_display = render_time_picker(
        "What time is it now?",
        default_hour=11,
        default_minute=0,
        default_period="PM",
    )

    # Question Block 2
    st.markdown(
        f"""
    <div style="background-color: {card_bg_blue}; border: 2px solid {card_border}; border-radius: 28px; padding: 24px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
        <div class="card-title">What time are you aiming to get up tomorrow?</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    target_hr, target_min, target_display = render_time_picker(
        "What time are you aiming to get up tomorrow?",
        default_hour=7,
        default_minute=0,
        default_period="AM",
    )

    # Question Block 3
    st.markdown(
        f"""
    <div style="background-color: {card_bg_purple}; border: 2px solid {card_border}; border-radius: 28px; padding: 24px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
        <div class="card-title">How much sleep are you aiming for? (7-9 hours of sleep is recommended; below 7 hours means sleep deprivation)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    aim_sleep = st.slider(
        "How much sleep are you aiming for? (7-9 hours of sleep is recommended; below 7 hours means sleep deprivation)",
        min_value=0.0,
        max_value=12.0,
        value=8.0,
        step=0.5,
        label_visibility="collapsed",
    )

    # Question Block 4
    st.markdown(
        f"""
    <div style="background-color: {card_bg_slate}; border: 2px solid {card_border}; border-radius: 28px; padding: 24px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
        <div class="card-title">(REQUIRED) Type in your rationale to delay sleep tonight (i.e. Why are you putting off sleep?)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    user_query = st.text_area(
        "(REQUIRED) Type in your rationale to delay sleep tonight (i.e. Why are you putting off sleep?)",
        placeholder="Type here...",
        height=120,
        label_visibility="collapsed",
    )

    if st.button(
        "SUBMIT RESPONSE to Generate Personalized Feedback", key="submit_mode_2"
    ):
        if not user_query.strip():
            st.error(
                "⚠️ **Input Required:** Please type your rationale in the box above before submitting."
            )
        else:
            t_now = datetime(2026, 1, 1, now_hr, now_min)
            t_wake = datetime(2026, 1, 1, target_hr, target_min)
            if t_wake <= t_now:
                t_wake += timedelta(days=1)

            available_sleep = (t_wake - t_now).total_seconds() / 3600.0

            st.info(
                f"⏱️ **Max Available Sleep Tonight:** **{available_sleep:.1f} hrs** (Target: **{aim_sleep} hrs**)"
            )

            if not openrouter_api_key:
                st.error(
                    "API Key not found. Please set `OPENROUTER_API_KEY` in Streamlit secrets."
                )
            else:
                with st.spinner(
                    "Executing retrieval & querying OpenRouter..."
                ):
                    top_matches = search_raw_text_chunks(
                        user_query, rag_chunks, top_k=3
                    )
                    context_str = "\n\n".join(
                        [f"Source ({m[2]}): {m[1]}" for m in top_matches]
                    )

                    system_prompt = f"""You are an accountability Sleep Coach dealing with bedtime procrastination. 
CRITICAL OUTPUT CONSTRAINTS:
1. Output MUST be between 1 and 3 sentences total. 
2. Output ONLY the final advice aimed at the user.
3. DO NOT include reasoning, chain-of-thought, meta-commentary, introductory text, or closing remarks. 
4. Never show inner logic or reference these system constraints in the output.


USER METRICS:
- Current Time: {now_display}
- Target Wake-Up Time: {target_display}
- Available Sleep Remaining: {available_sleep:.1f} hours
- User's Goal Sleep: {aim_sleep} hours

Address their delay rationale directly while contrasting their remaining available sleep ({available_sleep:.1f} hrs) against their target sleep goal ({aim_sleep} hrs). Provide supportive, persuasive advice in maximum 3 sentences based on the scientific context below.
Write a supportive answer in maximum 3 sentences.

CONTEXT:
{context_str}

USER NEGOTIATION RATIONALE:
{user_query}"""

                    try:
                        url = "https://openrouter.ai/api/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {openrouter_api_key}",
                            "Content-Type": "application/json",
                        }
                        payload = {
                            "model": "nvidia/nemotron-3.5-lightning:free",
                            "messages": [
                                {"role": "user", "content": system_prompt}
                            ],
                            "max_tokens": 1000,
                        }

                        response = requests.post(
                            url, headers=headers, json=payload, timeout=12
                        )
                        response.raise_for_status()
                        res_json = response.json()
                        raw_ai_response = res_json["choices"][0]["message"][
                            "content"
                        ]

                        final_response = clean_and_trim_response(
                            raw_ai_response
                        )

                        st.success("### AI Coach Guidance")
                        st.write(final_response)

                        with st.expander("🔍 View Retrieved Knowledge Context"):
                            seen_sources = set()
                            for match in top_matches:
                                source_name = match[2]
                                if source_name not in seen_sources:
                                    st.markdown(f"• **{source_name}**")
                                    seen_sources.add(source_name)
                    except Exception as e:
                        st.error(f"OpenRouter API Error: {e}")
