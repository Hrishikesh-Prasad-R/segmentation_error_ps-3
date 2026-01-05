# 🏆 The "Killer Demo" Script (IIT Hackathon Edition)

---

## 🚀 NEW FEATURE: Smart Dimension Selection
Before scoring, the agent **automatically detects** which dimensions apply based on the dataset schema.
- **Pitch:** "We don't blindly score all 7 dimensions. The agent reads the schema and decides what's relevant."
- **Example:** If no `date` column → "Timeliness" is skipped (marked N/A).
- **API Response:** Check `dimension_relevance` field in the result.

---

## Scenario 1: The "Invisible" Fraud (Showcasing ML)
**Pitch:** "Rules are dumb. A $999 transaction passes the <$1000 limit. But $999 for *coffee*? Our AI catches it."

**Action:** Upload `hero_fraud.csv`.
**Result:** Rules PASS, ML FAILS, GenAI explains it.

---

## Scenario 2: The "Cyber Attack" (Showcasing Security)
**Pitch:** "Most GenAI agents are easily tricked. Ours has Layer 2 Adversarial Detection."

**Action:** Upload `hero_attack.csv` (SQL injection payload).
**Result:** Layer 2 BLOCKS immediately. AI never sees bad data.

---

## Scenario 3: The "Pull The Plug" (Showcasing Resilience)
**Pitch:** "Watch what happens when the AI brain dies."

**Action:** Disconnect Wifi, upload `sample_payments.csv`.
**Result:** System degrades safely. Returns "SAFE_TO_USE (Rules Only)".

---

# 🧠 Judge Q&A Cheat Sheet

| Question | Answer |
|----------|--------|
| "Training stats?" | "Unsupervised. No pre-training. Fits on live data." |
| "Why not Neural Net?" | "Overkill. Isolation Forest is explainable." |
| "What if all data is bad?" | "Layer 4.1 Rules catch it before ML runs." |
| "Is this an LLM wrapper?" | "No. LLM is 5%. Core is 11-layer pipeline." |
| "How do you pick dimensions?" | "**NEW:** Auto-detect from schema. Skip N/A." |
