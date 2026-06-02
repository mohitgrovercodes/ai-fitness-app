# AI Fitness Gym — Multi-Agent Personal Training Platform

A production-grade, multi-agent personal training and diet orchestration platform built with a FastAPI backend and a Streamlit developer panel. The system features a 9-axis Biomechanical Injury Safety Engine, evidence-based textbook research retrieval, and self-learning VLM-based food analysis.

---

## 1. System Architecture

The application is structured into two main components: a core multi-agent backend and an interactive frontend visualization.

```
                  ┌────────────────────────────────────────┐
                  │          Streamlit UI Client           │
                  └───────────┬────────────────┬───────────┘
                              │                │
            JSON REST Payload │                │ Multipart / Image Upload
                              ▼                ▼
     ┌────────────────────────┴────────────────┴───────────┐
     │           FastAPI Application Backend               │
     └────────────────────────┬────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼ (Relational)       ▼ (Cache/Session)    ▼ (Embeddings/RAG)
   ┌───────────┐        ┌───────────┐        ┌───────────┐
   │ MySQL DB  │        │ Redis DB  │        │ ChromaDB  │
   └───────────┘        └───────────┘        └───────────┘
```

### Technology Stack
* **Core Logic & Graph Routing**: LangGraph, Pydantic, Python 3.11
* **Language Models**: `gpt-4o-mini` (via structured outputs & JSON schemas)
* **Databases**:
  * **MySQL**: Persists permanent user accounts, biometric profiles, onboarding preferences, and feedback logs.
  * **Redis**: Caches multi-turn chat session history, intermediate translations, and query hashes.
  * **ChromaDB**: Shared vector database storing embedded exercises and scientific textbooks.
* **Frontend**: Streamlit dev client with authentication, custom charts, and video players.

---

## 2. Multi-Agent Orchestration Layer

The backend uses a LangGraph execution graph to coordinate specialized fitness sub-agents:

```
                      [ User Input Query ]
                               │
                               ▼
               ┌───────────────────────────────┐
               │    Triage / Intake Node       │
               └───────────────┬───────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  Training    │    │  Nutrition   │    │  Textbook    │
    │  Specialist  │    │  Specialist  │    │  Research    │
    │    (CRAG)    │    │    (RAG)     │    │    (RAG)     │
    └───────┬──────┘    └───────┬──────┘    └───────┬──────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
               ┌───────────────────────────────┐
               │     Synthesis Node (Graph)    │
               └───────────────────────────────┘
```

1. **Triage / Intake Node**: Classifies user intents, parses physical injury constraints into a structured model, and handles multilingual detection and translation routing.
2. **Training Specialist (CRAG)**: Generates highly customized, multi-day workout routines. It loads pre-vetted RAG contexts, enforces strict exercise schemas, and splits planning into repeating microcycles (for $N > 12$ days) with progressive overload calculations performed by the backend.
3. **Nutrition Specialist**: Builds dietary programs matching Mifflin-St Jeor TDEE equations. Takes into account medical conditions, allergens, and dietary preferences (e.g. Vegan, Keto).
4. **Vision Agent**: Connects to the `/api/ai/chat-vision` endpoint. Performs self-learning calorie and macro estimations from uploaded meal photos using CLIP embeddings and VLM visual analysis.
5. **Textbook Q&A Agent**: Resolves scientific questions using a vector database of validated sports science textbooks, returning responses complete with citation footnotes.

---

## 3. The 9-Axis Biomechanical Injury Safety Engine

To protect injured users, the platform implements a hybrid, multi-tier safety pipeline that translates raw statements of pain (e.g., "left knee sprain") into concrete physical boundaries.

```
       [Raw User Injuries] ➔ [Intake LLM Classifies Constraints]
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ TIER 1: Post-Retrieval Deterministic Filter            │
        │ - Scans the 1,018-exercise production database.        │
        │ - Blocks movements violating any of the 9 joint axes:  │
        │   HIP, KNEE, ANKLE, LUMBAR_SPINE, THORACIC_SPINE,      │
        │   CERVICAL_SPINE, SHOULDER, ELBOW, WRIST               │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ TIER 2: Recency Safety Prompt Reinforcement            │
        │ - Formats pre-vetted exercise context with DB IDs.     │
        │ - Mandates strict literal boundaries to prevent drift. │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ TIER 3: LLM Biomechanical Safety Gate                  │
        │ - Pre-screens planned movements for stabilizing stress.│
        │ - Intercepts and replaces unsafe exercises.            │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ TIER 3 BACKSTOP: Deterministic Python Matcher          │
        │ - Hard stop verifying final timeline against rules.    │
        │ - Substitutes violations with group-aligned fallbacks. │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
                       [100% Safe Workout Output]
```

* **Exclusion Constraints**: Checks include joint movement blocking, kinetic chain load limitations (Open vs. Closed), spinal compression ratings, metabolic densities, and grip load caps.
* **Segment Coverage Refusal**: If a severe injury reduces the pool of safe exercises below a balanced threshold, the system intercepts the execution and refuses the query, advising the user to consult a physician.
* **Safe Recovery Matcher**: Replaces unsafe leaks with muscle-group-aligned, duplicate-free exercises from a curated recovery pool, updating descriptions with rehab cues and fuzzy-matching correct video files.

---

## 4. Getting Started

### Prerequisites
* Python 3.11+
* MySQL Server (running at port 3306)
* Redis (running at port 6379)
* Local GIF/Image dataset placed under `Data/exercises-dataset/`

### Installation & Run

1. **Clone and Configure Environment**:
   Create a `.env` file in the root directory based on standard credentials:
   ```env
   DATABASE_URL=mysql+pymysql://root:password@localhost:3306/fitnes_db
   REDIS_HOST=localhost
   REDIS_PORT=6379
   OPENAI_API_KEY=sk-proj-...
   TAVILY_API_KEY=tvly-...
   ```

2. **Start Backend Server**:
   ```bash
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

3. **Start Streamlit UI**:
   ```bash
   cd frontend
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
   Open `http://localhost:8501` to access the developer panel.