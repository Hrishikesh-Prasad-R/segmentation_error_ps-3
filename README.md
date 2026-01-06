# 🛡️ GenAI Data Quality Agent

<div align="center">

**🏆 IITM VISA Hackathon 2026 Solution 🏆**

*The First "Safe" GenAI Agent for Financial Data Quality*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange.svg)](https://ai.google.dev/)

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Tech Stack](#️-tech-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Demo Scenarios](#-demo-scenarios)
- [Team](#-team)

---

## 💡 Problem Statement

Payment organizations process **massive volumes of data**, but evaluating quality is:
- ⏳ **Manual** and time-consuming
- ❌ **Error-prone** with human oversight
- 🤖 **Risky with AI** due to hallucinations and lack of accountability

**Challenge**: Build a GenAI-powered data quality agent that is both intelligent AND trustworthy.

---

## 🚀 Our Solution

We built an **Agentic Data Quality System** that combines:
- ✨ **Creativity of GenAI** for intelligent insights
- 🔒 **Safety of Deterministic Rules** for reliability
- 📊 **Machine Learning** for anomaly detection
- ⚖️ **Accountability** through liability assignment

---

## ⭐ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Hybrid Intelligence** | Rules + ML + GenAI working together |
| 🛡️ **Safe Degradation** | Never crashes - falls back gracefully |
| 🔍 **Anomaly Detection** | Isolation Forest finds hidden patterns |
| ⚖️ **Liability Tracking** | Clear responsibility attribution |
| 📝 **Audit Trail** | Immutable logging of all decisions |
| 💡 **Actionable Insights** | GenAI-powered fix recommendations |

---

## 🏗️ Architecture

Our **12-Layer Pipeline** ensures safety at every step:

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT VALIDATION                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 1  │  Schema Validation      │  Rules               │
│  Layer 2  │  Column Detection       │  Rules               │
│  Layer 3  │  Feature Extraction     │  Rules               │
├─────────────────────────────────────────────────────────────┤
│                    HYBRID ANALYSIS                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 4  │  Hybrid Inference       │  Rules + ML          │
├─────────────────────────────────────────────────────────────┤
│                    OUTPUT STABILITY                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 5  │  Output Normalization   │  Logic               │
│  Layer 6  │  Output Validation      │  Logic               │
│  Layer 7  │  Conflict Resolution    │  Logic               │
│  Layer 8  │  Confidence Scoring     │  Logic               │
│  Layer 9  │  4-State Decision Gate  │  Logic               │
├─────────────────────────────────────────────────────────────┤
│                    ACCOUNTABILITY                           │
├─────────────────────────────────────────────────────────────┤
│  Layer 10 │  Liability Assignment   │  Legal               │
│  Layer 11 │  Immutable Audit Log    │  Audit               │
│  Layer 12 │  GenAI Summary          │  AI (Gemini)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI (Python) |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Machine Learning** | Scikit-Learn (Isolation Forest) |
| **AI/LLM** | Google Gemini Pro |
| **Data Processing** | Pandas, NumPy |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google Gemini API Key

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/segmentation_error_ps-3.git
cd segmentation_error_ps-3
```

### 2️⃣ Set Up Backend

```bash
# Navigate to backend
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Configure Environment

Create a `.env` file in the `backend` folder:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> 💡 **Get your API key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)

### 4️⃣ Run the Server

```bash
# From backend folder
python main.py
```

The server will start at `http://localhost:8000`

### 5️⃣ Access the Application

Open your browser and navigate to:

```
http://localhost:8000
```

### 6️⃣ Try It Out!

1. Upload a CSV file (try `sample_payments.csv` from the `backend` folder)
2. Watch the 12-layer analysis in action
3. Get actionable insights with fix suggestions

---

## 📁 Project Structure

```
segmentation_error_ps-3/
├── frontend/                    # Frontend application
│   └── index.html              # Main UI (single-page app)
│
├── backend/                     # Backend API server
│   ├── app/
│   │   ├── core/               # Core pipeline logic
│   │   │   ├── agent.py        # Main orchestrator
│   │   │   └── layers/         # 12 pipeline layers
│   │   │       ├── layer_1_input_schema.py
│   │   │       ├── layer_2_column_detection.py
│   │   │       ├── layer_3_feature_extraction.py
│   │   │       ├── layer_4_inference.py
│   │   │       ├── layer_5_output_normalization.py
│   │   │       ├── layer_6_output_validation.py
│   │   │       ├── layer_7_conflict_resolution.py
│   │   │       ├── layer_8_confidence.py
│   │   │       ├── layer_9_decision_gate.py
│   │   │       ├── layer_10_liability.py
│   │   │       ├── layer_11_logging.py
│   │   │       └── layer_12_final_summary.py
│   │   └── __init__.py
│   │
│   ├── datasets/               # Sample datasets
│   ├── tests/                  # Unit tests
│   ├── main.py                 # FastAPI entry point
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Docker configuration
│   └── .env.template           # Environment template
│
├── HACKATHON_DEMO.md           # Demo script for judges
└── README.md                   # This file
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the frontend application |
| `POST` | `/analyze` | Analyze uploaded data file |

### Analyze Endpoint

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_data.csv"
```

**Response:**
```json
{
  "composite_score": 85.5,
  "overall_status": "ACCEPT",
  "summary": "Data quality is acceptable...",
  "dimensions": [...],
  "anomalies": [...],
  "confidence_band": "HIGH",
  "liability_summary": "...",
  "next_steps": [...]
}
```

---

## 🧪 Demo Scenarios

See [`HACKATHON_DEMO.md`](./HACKATHON_DEMO.md) for detailed demo scripts:

1. **🔍 Invisible Fraud Detection** - ML catches what rules miss
2. **🛡️ SQL Injection Attack** - Adversarial input handling
3. **💪 System Resilience** - Graceful degradation demo
4. **📊 Borderline Cases** - Human-in-the-loop decisions

### Sample Files

| File | Purpose |
|------|---------|
| `hero_clean.csv` | High-quality data (score: 90+) |
| `hero_fraud.csv` | Contains hidden anomalies |
| `hero_attack.csv` | Adversarial SQL injection |
| `hero_borderline.csv` | Requires human review |

---

## 🎯 Why We're Different

| Others | Our Solution |
|--------|--------------|
| Pure AI (hallucinations) | **Rules ALWAYS override AI** |
| Black box decisions | **Full audit trail** |
| Crashes on bad input | **Safe degradation** |
| No accountability | **Liability assignment** |
| Generic insights | **Actionable fix commands** |

---

## 👥 Team

**Team Segmentation Error**

Built with ❤️ for IITM VISA Hackathon 2026

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**⭐ Star this repo if you found it useful! ⭐**

</div>
