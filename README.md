# 🌙 AI Sleep Coach MVP (RAG Only)

A pure Retrieval-Augmented Generation (RAG) conversational assistant grounded in evidence-based medical guidelines (CDC, NSF, Harvard, and NIH).

## 📌 Features
* **Mode 1: Morning Check-in & Habit Reflection** — Log sleep metrics and subjective alertness levels to receive personalized daytime coaching.
* **Mode 2: Bedtime Procrastination Coach** — Evaluate cognitive trade-offs and receive actionable advice when delaying sleep.
* **Subjective State Injection** — Uses a 1–9 alertness-sleepiness rating directly in LLM prompts to tailor recommendations.
* **Lightweight Keyword RAG Engine** — Matches queries against scientific sleep guidelines for context-grounded responses.

## 🛠️ Architecture
* **Frontend/UI:** Streamlit
* **Knowledge Retrieval:** Lightweight keyword overlap engine (`lightweight_rag_components.pkl`)
* **Inference Model:** OpenRouter API (`nvidia/nemotron-3.5-lightning:free`)

## 🚀 Setup & Deployment

### Local Execution
1. Clone or download the repository files.
2. Install dependencies:
   ```bash
   pip install streamlit joblib pandas numpy requests
