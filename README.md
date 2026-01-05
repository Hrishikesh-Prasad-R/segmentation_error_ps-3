# GenAI Data Quality Agent 🛡️🧠

> **The First "Safe" GenAI Agent for Financial Data Quality**
> Built for the IIT Madras AI Hackathon

## 💡 Problem Statement
Payment organizations process massive volumes of data, but evaluating quality is manual, slow, and error-prone. GenAI offers a solution but suffers from "hallucinations" and lack of accountability.

## 🚀 Our Solution
We built an **Agentic Data Quality System** that combines the creativity of GenAI with the safety of deterministic software.

### Key Differentiators
1.  **Rules > AI Architecture**: Deterministic contracts (Layer 1-4) *always* support or override AI opinions.
2.  **Unsupervised ML**: Uses `IsolationForest` (Scikit-Learn) to find complex anomalies without pre-training.
3.  **Safe Degradation**: The system never crashes. If the AI fails, it falls back to rules. If input is bad, it returns structured failure reports.
4.  **Liability Layer**: Explicitly attributes responsibility (System vs Human) for every decision.

## 🏗️ 11-Layer Architecture
| Layer | Function | Type |
|-------|----------|------|
| 1-3 | Input Validation & Features | Rules |
| 4 | **Hybrid Inference** (Rules + ML) | Hybrid |
| 5-8 | Output Stability & Conflict | Logic |
| 9 | **4-State Decision Gate** | Logic |
| 10 | Liability Assignment | Legal |
| 11 | Immutable Audit Log | Audit |
| 12 | **GenAI Summary** (Gemini) | AI |

## 🛠️ Tech Stack
- **Backend**: FastAPI
- **ML**: Scikit-Learn (Isolation Forest)
- **AI**: Google Gemini Pro
- **Data**: Pandas / NumPy

## 🏃‍♂️ Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run Server
```bash
python main.py
```

### 3. Analyze Data
Open `http://localhost:8000` and upload `sample_payments.csv`.

## 🧪 Demo Scenarios (For Judges)
See `HACKATHON_DEMO.md` for the full script showing:
1.  **Invisible Fraud** (ML Detection)
2.  **SQL Injection Attack** (Adversarial Defense)
3.  **System Resilience** (Safe Degradation)
