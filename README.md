# Social Support Eligibility System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green)](https://github.com/langchain-ai/langgraph) 

**AI-powered social support application processing system that reduces assessment time from 5-20 working days to under 2 minutes through multi-agent orchestration, hybrid ML/LLM reasoning, and multi-database architecture.**

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Database Schema](#-database-schema)
- [API Documentation](#-api-documentation)
- [Usage Examples](#-usage-examples)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Future Improvements](#-future-improvements)

---

## 🎯 Overview

### Problem Statement

Government social support departments face significant challenges:
- **Manual data entry** from scanned documents (data entry errors, 2-5 days)
- **Physical document collection** requiring applicant travel (1-3 days)
- **Semi-automated validation** with manual inconsistency checks (1-2 days)
- **Multiple review rounds** across departments (3-7 days)
- **Subjective decision-making** prone to bias

**Total processing time:** 5-20 working days per application

### Solution

An AI-powered system that:
1. **Ingests** 5 document types (Emirates ID, bank statement, resume, assets/liabilities, credit report)
2. **Extracts** data using OCR + LLM structuring
3. **Validates** cross-document consistency via graph database queries
4. **Scores** eligibility using ML classifier (deterministic, explainable)
5. **Recommends** financial support decisions + enablement programs (upskilling, job matching)

**Result:** 99% automated processing in ~2 minutes

---

## ✨ Key Features

### Multi-Agent Orchestration
- **4 specialized agents** (Extraction, Validation, Eligibility, Recommendation) orchestrated via LangGraph
- **ReAct reasoning** for goal-directed tool use
- **Reflexion pattern** for self-critique in validation

### Hybrid ML/LLM Architecture
- **scikit-learn HistGradientBoostingClassifier** for deterministic eligibility scoring
- **Phi-4-mini (3.8B)** locally-hosted LLM for language understanding
- **Feature importances** for explainability (government requirement)

### Multi-Database Design
- **PostgreSQL** - Structured data + audit trail (ACID transactions)
- **MongoDB** - Raw document storage (flexible schema)
- **Qdrant** - Vector similarity search (semantic program matching)
- **Neo4j** - Entity relationship graph (cross-document validation)

### Production-Quality Engineering
- **Graceful degradation** - System handles LLM failures, partial data extraction
- **Full observability** - Langfuse traces every agent, LLM call, classifier prediction
- **Audit trail** - Complete decision history for compliance
- **Resource-efficient** - Runs on 12GB RAM, 4GB VRAM

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Streamlit Chat UI (port 8501)               │
│                Interactive 3-stage workflow                 │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            FastAPI Gateway (port 8000)                      │
│  9 endpoints: health, CRUD, upload, process, status, result │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             LangGraph Agent Orchestration                   │
│               Linear Pipeline (v1)                          │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Agent 1: Data Extraction (ReAct)                │      │
│  │  - OCR (Tesseract) + PDF/Excel parsing           │      │
│  │  - LLM structuring (Phi-4-mini via Ollama)      │      │
│  │  Output: Structured JSON for all 5 documents     │      │
│  └────────────┬─────────────────────────────────────┘      │
│               │                                             │
│  ┌────────────▼─────────────────────────────────────┐      │
│  │  Agent 2: Data Validation (Reflexion)            │      │
│  │  - Cross-document consistency checks              │      │
│  │  - Neo4j graph queries (address, employer)       │      │
│  │  - PostgreSQL queries (income matching)          │      │
│  │  - Self-critique loop                            │      │
│  │  Output: List of inconsistencies with severity   │      │
│  └────────────┬─────────────────────────────────────┘      │
│               │                                             │
│  ┌────────────▼─────────────────────────────────────┐      │
│  │  Agent 3: Eligibility Scoring (ReAct + ML)       │      │
│  │  - Feature engineering (13 features)             │      │
│  │  - scikit-learn HistGradientBoosting Classifier  │      │
│  │  - Feature importances extraction                │      │
│  │  - LLM interprets edge cases                     │      │
│  │  Output: Category + confidence + reasoning       │      │
│  └────────────┬─────────────────────────────────────┘      │
│               │                                             │
│  ┌────────────▼─────────────────────────────────────┐      │
│  │  Agent 4: Recommendation (ReAct)                 │      │
│  │  - Generate approval/decline reasoning            │      │
│  │  - Qdrant vector similarity search               │      │
│  │  - Semantic program matching (resume ↔ programs) │      │
│  │  - Fallback to keyword matching                  │      │
│  │  Output: Decision + 3-5 enablement programs      │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                │
│  ┌────────────┬────────────┬────────────┬──────────────┐   │
│  │ PostgreSQL │  MongoDB   │  Qdrant    │   Neo4j      │   │
│  │ (5432)     │  (27017)   │ (6333/34)  │  (7474/7687) │   │
│  │            │            │            │              │   │
│  │ Structured │    Raw     │  Vector    │    Graph     │   │
│  │ data +     │  document  │ similarity │ relationships│   │
│  │ audit trail│  storage   │  search    │ for validation│  │
│  └────────────┴────────────┴────────────┴──────────────┘   │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             Langfuse Observability (port 3000)              │
│        Traces every agent, LLM call, classifier run         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Technologies (Mandated by Assessment)

| Category | Tool | Justification |
|---|---|---|
| **Orchestration** | LangGraph | Explicit state graphs, checkpointing, streaming |
| **LLM** | Phi-4-mini (3.8B) via Ollama | Fits 4GB VRAM, strong JSON output, tool-calling |
| **ML** | scikit-learn HistGradientBoostingClassifier | Handles mixed features, native categorical support, explainable |
| **Databases** | PostgreSQL, MongoDB, Qdrant, Neo4j | Each serves distinct query pattern (see Database Schema) |
| **API** | FastAPI | Async support, automatic docs, Pydantic validation |
| **UI** | Streamlit | Rapid prototyping, real-time updates, chat interface |
| **Observability** | Langfuse v2 | End-to-end tracing, prompt management |
| **OCR** | Tesseract | Zero VRAM, sufficient for machine-generated docs |
| **Embeddings** | sentence-transformers | CPU-based, 384-dim, zero VRAM |

### Key Design Decisions

**Why 4 Databases?**


Each database serves a specific purpose that others can't efficiently handle:

1. **PostgreSQL** - ACID transactions, foreign keys for audit trail
   ```sql
   -- One query reconstructs complete application history
   SELECT * FROM decisions d
   JOIN eligibility_scores e ON d.application_id = e.application_id
   WHERE applicant_id = 'xxx';
   ```

2. **MongoDB** - Flexible schema for varying document structures
   ```javascript
   // Bank statement has transactions array, credit report doesn't
   db.extractions.find({document_type: "bank_statement"})
   ```

3. **Qdrant** - Vector similarity for semantic matching
   ```python
   # Find programs similar to resume without keyword overlap
   qdrant.search(collection="programs", query_vector=resume_embedding)
   ```

4. **Neo4j** - Relationship traversal for validation
   ```cypher
   // Find address mismatches across documents
   MATCH (a:Applicant)-[:LIVES_AT]->(addr1)<-[:REPORTED_IN]-(doc1),
         (a)-[:LIVES_AT]->(addr2)<-[:REPORTED_IN]-(doc2)
   WHERE addr1.emirate <> addr2.emirate
   RETURN a.name, doc1.type, doc2.type
   ```

**Why Hybrid ML/LLM?**

| Task | Tool | Why |
|---|---|---|
| Document parsing | LLM | Language understanding required |
| Eligibility scoring | Classifier | Deterministic, explainable (government requirement) |
| Reasoning generation | LLM | Natural language output required |

---

## 💻 System Requirements

- **CPU:** Intel i5 or AMD Ryzen 5 (4+ cores recommended)
- **RAM:** 12 GB minimum (system reserves ~1.5-2 GB)
- **GPU:** NVIDIA RTX 2050 (4GB VRAM) or better (optional but recommended)
- **Storage:** 10 GB free space
- **OS:** Windows 10/11, Linux, or macOS
- **Docker:** Desktop with 6 GB memory limit configured
- **Python:** 3.11 or higher

### Resource Usage Breakdown

```
Total 12 GB RAM:
├── Windows OS: ~3-4 GB
├── Docker containers (4 DBs + Langfuse): ~3 GB
├── Ollama + Phi-4-mini: ~2.5 GB
├── Python (FastAPI + Streamlit): ~0.5-1 GB
└── Headroom: ~1-2 GB
```

---

## 🚀 Quick Start

### 1. Prerequisites

**Install required software:**

```powershell
# Docker Desktop (with 6 GB memory limit in settings)
winget install Docker.DockerDesktop

# Python 3.11+
winget install Python.Python.3.11

# Tesseract OCR
winget install UB-Mannheim.TesseractOCR

# Ollama
irm https://ollama.com/install.ps1 | iex
```

**Pull the LLM model:**
```powershell
ollama pull phi4-mini
```

### 2. Clone Repository

```powershell
git clone https://github.com/ahd-rafi/Social-Support-Eligibility-System.git
cd Social-Support-Eligibility-System
```

### 3. Setup Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Configure Environment

```powershell
# Copy example environment file
cp .env.example .env

# Edit .env with your passwords:
# - POSTGRES_PASSWORD
# - MONGO_INITDB_ROOT_PASSWORD
# - NEO4J_PASSWORD
# (See .env.example for all variables)
```

### 5. Start Databases

```powershell
# Development mode (databases only, ~3 GB RAM)
docker compose -f docker-compose.dev.yml up -d

# Full stack mode (with Langfuse, ~4.5 GB RAM)
# docker compose up -d

# Wait for all services to be healthy
docker compose -f docker-compose.dev.yml ps
```

### 6. Initialize System

```powershell
# Create database schemas
python -m src.db.init_schema

# Generate synthetic test data
python -m data.synthetic.generate

# Train ML classifier
python -m scripts.train_classifier

# Seed databases with programs and policies
python -m data.seed

# Verify setup
python -m scripts.verify_databases
```

### 7. Run Application

```powershell
# Terminal 1: Start API
uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Start UI
streamlit run frontend/app.py --server.port 8501
```

### 8. Test the System

1. Open http://localhost:8501
2. Upload **sample documents** from `data/synthetic/output/sample_application/`:
   - emirates_id.png
   - bank_statement.pdf
   - resume.pdf
   - assets_liabilities.xlsx
   - credit_report.pdf
3. Click "Start Processing"
4. View results in ~30-60 seconds

**Note:** Sample application data is included in the repository for testing.

---

## 📁 Project Structure

```
Social-Support-Eligibility-System/
├── src/
│   ├── agents/              # LangGraph agents
│   │   ├── extraction.py    # Document parsing (ReAct)
│   │   ├── validation.py    # Cross-doc checks (Reflexion)
│   │   ├── eligibility.py   # ML scoring (ReAct + classifier)
│   │   ├── recommendation.py # Final decision (ReAct)
│   │   ├── state.py         # LangGraph state schema
│   │   ├── graph.py         # Agent orchestration
│   │   └── llm.py           # LLM initialization
│   ├── ml/                  # Machine learning
│   │   ├── classifier.py    # HistGradientBoosting classifier
│   │   ├── embeddings.py    # sentence-transformers wrapper
│   │   └── features.py      # Feature engineering
│   ├── parsers/             # Document processing
│   │   ├── ocr.py           # Tesseract wrapper + preprocessing
│   │   ├── pdf.py           # PDF text extraction
│   │   ├── router.py        # Document type detection
│   │   ├── bank_statement.py
│   │   ├── resume.py
│   │   ├── credit_report.py
│   │   └── assets_liabilities.py
│   ├── db/                  # Database clients
│   │   ├── postgres.py      # PostgreSQL async client
│   │   ├── mongo.py         # MongoDB client
│   │   ├── qdrant_client.py # Qdrant wrapper
│   │   ├── neo4j_client.py  # Neo4j driver wrapper
│   │   └── init_schema.py   # Schema creation script
│   ├── api/                 # FastAPI application
│   │   ├── main.py          # API entry point
│   │   ├── routes.py        # Endpoint definitions
│   │   └── schemas.py       # Pydantic models
│   ├── observability/       # Langfuse integration
│   │   └── tracing.py       # Trace decorators
│   └── config.py            # Pydantic settings
├── frontend/
│   └── app.py               # Streamlit UI (3-stage workflow)
├── data/
│   ├── synthetic/
│   │   ├── generate.py      # Profile generation
│   │   ├── generate_documents.py # PDF/Excel creation
│   │   └── output/
│   │       └── sample_application/  # Demo documents
│   ├── programs/            # Enablement program descriptions
│   ├── policies/            # Policy documents (RAG)
│   └── seed.py              # Database seeding script
├── models/                  # Trained classifier artifacts
│   └── .gitkeep
├── scripts/
│   ├── train_classifier.py  # ML model training
│   ├── verify_databases.py  # Connection verification
│   └── clear_database.py    # Reset for testing
├── tests/                   # Integration tests
│   ├── test_e2e.py          # End-to-end workflow
│   ├── test_agents.py       # Individual agent tests
│   └── test_parsers.py      # Document parsing tests
├── docker-compose.dev.yml   # Development stack (DBs only)
├── docker-compose.yml       # Full stack (DBs + Langfuse)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
├── .gitignore              # Git exclusions
└── README.md               # This file
```

---

## 🗄️ Database Schema

### PostgreSQL Tables

**Core Entities:**
```sql
applicants          -- Structured applicant profiles
applications        -- Application records with status tracking
extraction_results  -- Parsed document data
validation_results  -- Detected inconsistencies
applicant_features  -- 13-feature vectors for ML
eligibility_scores  -- Classifier predictions + importances
decisions           -- Final recommendations (audit trail)
audit_log          -- System events
```

**Key Relationships:**
```sql
applicants (1) → (N) applications → (1) decisions
applications (1) → (N) extraction_results
applications (1) → (N) validation_results
```

### MongoDB Collections

```javascript
raw_documents    // Uploaded file metadata
extractions      // Raw extraction output with versioning
```

### Qdrant Collections

```python
applicant_profiles      // Resume embeddings (384-dim)
enablement_programs     // Program description embeddings
policy_documents        // Policy text embeddings
```

### Neo4j Graph

```cypher
(:Person {id, name, emirates_id})
(:Employer {name})
(:Address {street, emirate})
(:Document {type, application_id})

-[:WORKS_AT]->
-[:LIVES_AT]->
-[:FAMILY_OF]->
-[:REPORTED_IN]->
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "databases": {
    "postgres": "connected",
    "mongodb": "connected",
    "qdrant": "connected",
    "neo4j": "connected"
  }
}
```

#### Create Application
```http
POST /applications
Content-Type: application/json

{
  "applicant_name": "Ahmed Al-Mansoori",
  "household_size": 5
}
```

**Response:**
```json
{
  "application_id": "uuid",
  "application_ref": "APP-2026-001234",
  "status": "pending"
}
```

#### Upload Document
```http
POST /applications/{application_id}/upload
Content-Type: multipart/form-data

file: <binary>
document_type: "emirates_id" | "bank_statement" | "resume" | "assets_liabilities" | "credit_report"
```

#### Start Processing
```http
POST /applications/{application_id}/process
```

#### Get Status (Polling)
```http
GET /applications/{application_id}/status
```

**Response:**
```json
{
  "application_id": "uuid",
  "processing_status": "extracting" | "validating" | "scoring" | "recommending" | "complete" | "error",
  "current_agent": "extraction",
  "extraction_complete": true,
  "validation_complete": false,
  "eligibility_complete": false,
  "recommendation_complete": false,
  "error": null
}
```

#### Get Result
```http
GET /applications/{application_id}/result
```

**Response:**
```json
{
  "application_id": "uuid",
  "eligibility_category": "approved_financial",
  "eligibility_score": 0.92,
  "recommendation": {
    "decision": "approved",
    "reasoning": "Applicant qualifies based on low income...",
    "key_factors": ["monthly_income", "household_size"],
    "confidence": 0.93
  },
  "enablement_programs": [
    {
      "program_name": "Financial Literacy Workshop",
      "description": "...",
      "match_score": 0.85
    }
  ],
  "validation_issues": [
    {
      "severity": "INFO",
      "description": "Address differs between documents"
    }
  ]
}
```

---

## 📖 Usage Examples

### Python API Client

```python
import requests
import time
from pathlib import Path

API_BASE = "http://localhost:8000/api"

# 1. Create application
response = requests.post(f"{API_BASE}/applications", json={
    "applicant_name": "Test Applicant",
    "household_size": 4
})
app_id = response.json()["application_id"]
print(f"Created: {app_id}")

# 2. Upload documents
docs = [
    ("emirates_id.png", "emirates_id"),
    ("bank_statement.pdf", "bank_statement"),
    ("resume.pdf", "resume"),
    ("assets_liabilities.xlsx", "assets_liabilities"),
    ("credit_report.pdf", "credit_report"),
]

for file_path, doc_type in docs:
    with open(f"data/synthetic/output/sample_application/{file_path}", "rb") as f:
        requests.post(
            f"{API_BASE}/applications/{app_id}/upload",
            files={"file": f},
            data={"document_type": doc_type}
        )
    print(f"Uploaded: {file_path}")

# 3. Start processing
requests.post(f"{API_BASE}/applications/{app_id}/process")
print("Processing started...")

# 4. Poll for completion
while True:
    status = requests.get(f"{API_BASE}/applications/{app_id}/status").json()
    print(f"Status: {status['processing_status']} ({status['current_agent']})")
    
    if status["processing_status"] == "complete":
        break
    elif status["processing_status"] == "error":
        print(f"Error: {status['error']}")
        break
    
    time.sleep(2)

# 5. Get final result
result = requests.get(f"{API_BASE}/applications/{app_id}/result").json()
print(f"\n✓ Decision: {result['recommendation']['decision']}")
print(f"✓ Confidence: {result['recommendation']['confidence']:.0%}")
print(f"✓ Programs: {len(result['enablement_programs'])}")
```

---

## 🧪 Testing

### Run All Tests
```powershell
pytest tests/ -v
```

### Run Specific Test Suite
```powershell
# End-to-end workflow
pytest tests/test_e2e.py -v

# Individual agents
pytest tests/test_agents.py -v

# Document parsers
pytest tests/test_parsers.py -v
```

### Manual Testing
```powershell
# Clear databases
python -m scripts.clear_database

# Upload sample documents via UI
# http://localhost:8501
```

---

## 🚢 Deployment

### Local Development
See [Quick Start](#-quick-start) above.

### Production Considerations

**1. Database Scaling:**
```yaml
# Use managed services
PostgreSQL: AWS RDS, Azure Database
MongoDB: MongoDB Atlas
Qdrant: Qdrant Cloud
Neo4j: Neo4j Aura
```

**2. Model Serving:**
```yaml
# Distributed inference
- Deploy Ollama on dedicated GPU servers
- Use model quantization (Q4_K_M → Q8 for better quality)
- Consider vLLM for batching + higher throughput
```

**3. API Scaling:**
```yaml
# Horizontal scaling with load balancer
- Multiple FastAPI instances behind nginx
- Celery for background task queue
- Redis for caching + session management
```

**4. Monitoring:**
```yaml
# Production observability
- Langfuse for AI traces
- Prometheus + Grafana for metrics
- ELK stack for logs
- Sentry for error tracking
```

**5. Security:**
```yaml
# Hardening
- Enable SSL/TLS (HTTPS)
- API key authentication
- Rate limiting
- Input sanitization
- Database connection pooling with SSL
- Secrets management (AWS Secrets Manager, Azure Key Vault)
```

---

## 🔮 Future Improvements

### Short-term (Next Sprint)
- [ ] **Arabic language support** (bilingual UI + Arabic LLM)
- [ ] **Batch processing** (handle multiple applications concurrently)
- [ ] **Advanced RAG** (policy document retrieval for edge cases)
- [ ] **Webhook notifications** (status updates via email/SMS)

### Medium-term (Next Quarter)
- [ ] **Distributed training** (federated learning across departments)
- [ ] **Model fine-tuning** (domain-specific Phi-4-mini on government data)
- [ ] **Event-driven architecture** (Kafka for real-time processing)
- [ ] **A/B testing framework** (compare model versions)
- [ ] **Bias detection** (automated fairness metrics)

### Long-term (Roadmap)
- [ ] **Multi-tenancy** (support multiple government departments)
- [ ] **Explainable AI dashboard** (visualize decision factors for case workers)
- [ ] **Handwritten document OCR** (PaddleOCR or Google Vision AI)
- [ ] **Voice interface** (Arabic speech-to-text for phone applications)
- [ ] **Blockchain audit trail** (immutable decision history)

---
