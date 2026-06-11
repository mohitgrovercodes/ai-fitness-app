<div align="center">

# 🏋️ Agentic AI Gym

### Multi-Agent Fitness & Nutrition Platform

*A production-grade agentic AI system that turns natural-language fitness goals into personalized workout plans and meal plans — with multi-turn conversation, self-learning food vision, multilingual support, and a deterministic biomechanical safety layer in development.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-state%20machine-FF6F00)
![LangChain](https://img.shields.io/badge/LangChain-orchestration-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6F61)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-EE4C2C)
![Redis](https://img.shields.io/badge/Redis-cache%20%2B%20memory-DC382D?logo=redis&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-profiles%20%2B%20auth-4479A1?logo=mysql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dev%20UI-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-deployable-2496ED?logo=docker&logoColor=white)

</div>

---

## 🎯 What It Does

Tell it *"I want to lose 5kg, I'm vegetarian with a peanut allergy, and I've got a sore knee — give me a 7-day plan"* — and get back:

- A **knee-safe 7-day workout split** with exercise GIFs, sets/reps progression, and rest-day programming
- A **vegetarian, peanut-free meal plan** with daily macro totals, calorie targets matched to your goal, and 4-4-9 macro-math validation
- All written in **English, Hindi, or Hinglish**, depending on how you asked

Upload a photo of a meal and ask *"should I eat this?"* — the **self-learning Vision Agent** identifies the dish, looks up its nutrition (or estimates it from the visible portion), and tells you whether it fits your goal.

Ask *"how have I been progressing?"* — the **Progress Agent** reads your conversation history, parses every workout and meal you've discussed, and gives you a personalized journey summary.

---

## ✨ Highlights

| | |
|---|---|
| 🧠 **Multi-Agent Orchestration** | 8-node LangGraph state machine running 5 specialist agents in parallel via `asyncio.gather` |
| 🔍 **Adaptive RAG** | 3-phase pipeline (profile-enriched semantic search → multi-query expansion → Tavily web fallback) over 39,358 food items and 2,800+ exercises |
| 👁️ **Self-Learning Vision** | 4-tier CLIP→VLM decision engine; newly identified dishes are stored as moving-average centroids in ChromaDB, so the next sighting hits Tier-1 CLIP at zero API cost |
| 🛡️ **Dual Safety Guardrails** | Input safety check + output safety check, both with multilingual policy responses |
| 🌐 **Trilingual Native I/O** | English / Hindi (Devanagari) / Hinglish (Roman script) with LLM-driven translation at both input and structured output layers |
| 📊 **Code-Level Validators** | 4-4-9 macro-calorie math, protein floor (1.6g/kg), fat cap (≤35%), calorie density physics checks — none of it depends on the LLM being right |
| 🔐 **Production Auth** | JWT with DB-verified sessions, password-confirmed account deletion, structured error handling |
| 💾 **Persistent Memory** | Redis-backed rolling conversation memory with 20-message cursor-based summarization |
| 📈 **Feedback Loop** | Thumbs-up/down per response, admin analytics dashboard, per-intent satisfaction tracking |
| 🦴 **Biomechanical Safety** *(in development)* | 7-feature exercise tagging + deterministic Python filter + constrained-decoding output schema — replaces probabilistic keyword backstops with a structural guarantee |

---

## 🏗️ Architecture

The system runs on a LangGraph state machine. Every user message flows through safety → orchestration → parallel specialists → synthesis → output safety, with persistent memory and structured logging at every node.

```mermaid
graph TD
    U[👤 User] -->|message + optional image| API[FastAPI Endpoint]
    API --> SG1[🛡️ Safety Guardrail<br/>input check]
    SG1 -->|safe| ORCH[🎯 Orchestrator<br/>intent classification<br/>+ language detection]
    SG1 -->|unsafe| RESP[Polite Refusal]
    ORCH --> ROUT[🔀 Agent Router]
    ROUT --> COORD[⚡ Specialists Coordinator<br/>parallel execution]
    COORD --> A1[🏋️ Training Agent]
    COORD --> A2[🥗 Nutrition Agent]
    COORD --> A3[👁️ Vision Agent]
    COORD --> A4[📈 Progress Agent]
    COORD --> A5[📚 Domain Agent]
    A1 --> SYN[✨ Synthesis Layer<br/>Master Coach LLM]
    A2 --> SYN
    A3 --> SYN
    A4 --> SYN
    A5 --> SYN
    SYN --> SG2[🛡️ Output Safety]
    SG2 --> MEM[💾 Memory Manager<br/>Redis history + summary]
    MEM --> RESP_OK[🎁 Final Response<br/>text + workout + meals + GIFs]

    style SG1 fill:#fee,stroke:#c00
    style SG2 fill:#fee,stroke:#c00
    style ORCH fill:#fef,stroke:#609
    style COORD fill:#eff,stroke:#069
    style SYN fill:#efe,stroke:#060
```

---

## 🤖 The Multi-Agent System

Each specialist is a self-contained agent with its own RAG pipeline, prompt template, structured-output schema, and code-level validators.

| Agent | Specialty | Backend tools |
|---|---|---|
| 🏋️ **Training Agent** | Workout plans, exercise selection, multi-day splits with progressive overload. Dynamic cycle expansion for plans > 6 days. Activity-level-aware intensity scaling. | ChromaDB (2,800+ exercises), media matcher for GIFs/images, web fallback |
| 🥗 **Nutrition Agent** | Meal plans, macro math, calorie targeting, dietary preference enforcement, chunked generation for multi-day plans. | ChromaDB (39,358 foods), web fallback, 4-4-9 macro validator, protein floor + fat cap enforcers |
| 👁️ **Vision Agent** | Food image identification, portion estimation, self-learning DB writes. 4 decision tiers based on CLIP confidence + ambiguity gap. | CLIP ViT-B/32 (local), GPT-4o-mini Vision (fallback), ChromaDB centroid updates |
| 📈 **Progress Agent** | Personalized journey reports from Redis chat history, parsing past meals and workouts into a structured journal. | Redis history reader, profile-aware LLM synthesis |
| 📚 **Domain Agent** | Evidence-based fitness science Q&A (anatomy, BMR, hypertrophy, recovery), with web fallback for up-to-date research. | Tavily web search + LLM reasoning |

### The Orchestrator's Cleverness

- **Multi-intent classification**: a single message can fire multiple agents in parallel. Body-transformation goals auto-include both `workout` and `nutrition`.
- **Language detection at the boundary**: Hindi (Devanagari) and Hinglish (Roman-script Hindi) are detected upfront, translated to English for RAG, then re-rendered in the user's script at synthesis time.
- **Image-aware routing**: when `image_bytes` is present, the orchestrator hands off to the Vision Agent and strips redundant nutrition intents that would compete with the image analysis.

---

## 🛡️ Biomechanical Safety (Prototype → v1.0 in progress)

The legacy three-tier safety pipeline (keyword excluder + LLM safety review + Python backstop) is being replaced with a **Vetted Exercise Manifold**:

1. **Pre-tagged library** — every exercise gets 7 closed-vocabulary biomechanical tags (joints involved, kinetic chain, axial compression, grip demand, impact, upper-limb support, metabolic density).
2. **Structured intake** — user injury text is translated to an `InjuryConstraint` via Pydantic structured output. Closed-vocabulary enums mean hallucinated body parts get rejected at the JSON-schema layer.
3. **Deterministic Python filter** — operates on enum set intersection and ordinal comparison. Zero string matching. The legacy "block 'squat' → also blocks 'leg raise'" bug class is structurally impossible.
4. **Constrained decoding** — the workout-generating LLM's structured output schema is `Literal[*safe_pool_ids]`. The model **physically cannot** emit an exercise outside the deterministically-filtered safe pool.
5. **Refusal as a first-class outcome** — segment-coverage threshold detects when no balanced plan is possible and escalates to "consult a physiotherapist."

📄 Full design: [`app/safety/README.md`](app/safety/README.md) → architectural overview & validated test scenarios
📄 Roadmap: [`app/safety/IMPLEMENTATION_PLAN.md`](app/safety/IMPLEMENTATION_PLAN.md) → 12-phase plan with risk register & production rollout

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- MySQL 8.x running locally (or accessible)
- Redis (optional but recommended for conversation memory)
- An OpenAI API key
- A Tavily API key (optional, for web search fallback)

### 1. Backend setup

```bash
# Clone + enter
git clone <your-repo-url>
cd ai-fitness-app

# Install dependencies
pip install -r requirements.txt

# Configure secrets (NEVER commit this file)
cat > .env <<EOF
OPENAI_API_KEY="sk-..."
TAVILY_API_KEY="tvly-..."          # optional
MYSQL_URL="mysql+pymysql://root:password@localhost:3306/fitness_db"
REDIS_URL="redis://localhost:6379/0"
SECRET_KEY="$(openssl rand -hex 32)"
EOF

# Run the API
uvicorn app.main:app --reload
```

Health check: `curl http://localhost:8000/health` → `{"status": "ok", "components": {...}}`

### 2. Frontend (Streamlit dev visualization)

In a second terminal:

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` → register an account → onboarding form → start chatting with your AI coach.

📄 Frontend docs: [`frontend/README.md`](frontend/README.md)

### 3. Docker (optional)

```bash
docker compose up
```

A `.dockerignore` keeps `.env`, `venv/`, the ChromaDB store, and the local dataset out of the image.

---

## 📦 Tech Stack

<table>
<tr><th>Layer</th><th>Tech</th><th>What it does</th></tr>
<tr><td><b>API</b></td><td>FastAPI, Uvicorn, Pydantic</td><td>Async HTTP, request/response validation</td></tr>
<tr><td><b>Orchestration</b></td><td>LangGraph, LangChain</td><td>State machine, parallel agents, structured-output LLM calls</td></tr>
<tr><td><b>LLMs</b></td><td>OpenAI GPT-4o-mini</td><td>All specialist agents, intent classification, synthesis, vision fallback</td></tr>
<tr><td><b>Vision</b></td><td>OpenAI CLIP ViT-B/32 + GPT-4o-mini Vision</td><td>Local image matching + OOD fallback + self-learning</td></tr>
<tr><td><b>Vector DB</b></td><td>ChromaDB</td><td>Food + exercise embeddings, image centroids, RAG retrieval</td></tr>
<tr><td><b>Embeddings</b></td><td>OpenAI text-embedding-3-small (1536-d)</td><td>Semantic search over 39K foods + 2,800 exercises</td></tr>
<tr><td><b>Web Search</b></td><td>Tavily</td><td>Adaptive RAG Phase-3 fallback</td></tr>
<tr><td><b>Profile / Auth DB</b></td><td>MySQL + SQLAlchemy</td><td>Users, profiles, feedback</td></tr>
<tr><td><b>Cache + Memory</b></td><td>Redis</td><td>Rolling chat history, conversation summaries</td></tr>
<tr><td><b>Auth</b></td><td>JWT (python-jose) + bcrypt (passlib)</td><td>DB-verified sessions, password re-auth on delete</td></tr>
<tr><td><b>Frontend</b></td><td>Streamlit + requests</td><td>Developer visualization, multi-page app, native chat primitives</td></tr>
<tr><td><b>Deployment</b></td><td>Docker + docker-compose</td><td>Single-command local stack</td></tr>
</table>

---

## 🔌 API Reference

All endpoints require `Authorization: Bearer <jwt>` unless noted.

<details>
<summary><b>🔐 Auth — /api/auth/*</b></summary>

| Method | Path | Description | Auth |
|---|---|---|---|
| `POST` | `/register` | Create a new user account | ❌ |
| `POST` | `/login` | Authenticate and receive a JWT | ❌ |
| `DELETE` | `/account` | Permanently delete account (password re-confirmation in body) | ✅ |

</details>

<details>
<summary><b>👤 Profile — /api/profile/*</b></summary>

| Method | Path | Description |
|---|---|---|
| `GET` | `/me` | Get the authenticated user's profile |
| `POST` | `/onboarding` | First-time profile creation |
| `PATCH` | `/me` | Update profile fields |

</details>

<details>
<summary><b>🤖 AI — /api/ai/*</b></summary>

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Multi-turn chat through the full agent graph |
| `POST` | `/chat-vision` | Chat with image upload (multipart/form-data) → Vision Agent |
| `POST` | `/generate-workout` | Direct Training Agent call with form-style body |
| `POST` | `/generate-diet` | Direct Nutrition Agent call |
| `POST` | `/ask-domain` | Direct Domain Agent call for science questions |

</details>

<details>
<summary><b>💬 Feedback — /api/feedback/*</b></summary>

| Method | Path | Description |
|---|---|---|
| `POST` | `/submit` | Submit thumbs-up/down on an AI response |
| `GET` | `/history` | My recent feedback entries |
| `GET` | `/summary` | My satisfaction rate + recent comments |
| `GET` | `/admin/summary` | 🔒 Global satisfaction (admin only) |
| `GET` | `/admin/list` | 🔒 Paginated all-user feedback (admin only) |

</details>

<details>
<summary><b>🩺 Health — /ping & /health</b></summary>

| Method | Path | Description |
|---|---|---|
| `GET` | `/ping` | Fast liveness check (process up?) |
| `GET` | `/health` | Deep readiness check (MySQL, ChromaDB, Redis status) |

</details>

---

## 📂 Project Structure

```
ai-fitness-app/
├── app/                            # FastAPI backend
│   ├── main.py                     # entrypoint + lifespan + middleware
│   ├── agents/                     # 5 specialist agents + orchestrator + router
│   │   ├── orchestrator.py
│   │   ├── router.py
│   │   ├── training_agent.py
│   │   ├── nutrition_agent.py
│   │   ├── vision_agent.py
│   │   ├── progress_agent.py
│   │   └── domain_agent.py
│   ├── core/                       # LangGraph wiring, DBs, security, state
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── database.py             # ChromaDB singleton manager
│   │   ├── sql_db.py               # MySQL engine + session
│   │   ├── redis_client.py
│   │   ├── security.py             # JWT + DB-verified auth dependency
│   │   └── config.py
│   ├── modules/                    # API routes grouped by domain
│   │   ├── auth/                   # register, login, delete-account
│   │   ├── profile/                # onboarding, edit, me
│   │   ├── ai/                     # chat, vision, direct generators
│   │   └── feedback/               # submit, history, summary, admin
│   ├── tools/                      # RAG + web search + vision tools
│   ├── embeddings/                 # one-shot scripts that populate ChromaDB
│   ├── schema/                     # cross-module Pydantic models
│   ├── safety/                     # ⭐ biomechanical safety prototype
│   │   ├── schema.py
│   │   ├── filter.py
│   │   ├── intake.py
│   │   ├── tags_lower_body.json
│   │   ├── README.md
│   │   └── IMPLEMENTATION_PLAN.md
│   ├── common/                     # response helpers
│   └── utils/                      # logger, gif matcher
├── frontend/                       # Streamlit dev visualization
│   ├── streamlit_app.py            # entry: login gate + dashboard
│   ├── lib/                        # api_client, auth, renderers, debug
│   ├── pages/                      # Chat, Profile, Account (more coming)
│   ├── .streamlit/                 # config + secrets.toml.example
│   └── README.md
├── Data/                           # exercise + food datasets (gitignored)
├── chromadb_store/                 # vector store (gitignored)
├── docker/                         # Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── requirements.txt
```

---

## 🗺️ Roadmap

### ✅ Shipped

- Multi-agent backend with LangGraph orchestration
- Adaptive RAG over food + exercise libraries
- Self-learning Vision Agent
- Trilingual I/O
- JWT auth with DB-verified sessions + delete account
- Redis-backed conversation memory
- Centralized TDEE / calorie-target computation across all endpoints
- `/ping` + `/health` readiness probes
- Streamlit frontend Phases 0–2: auth + profile + multi-turn chat with structured renderers
- Biomechanical safety **prototype** with deterministic filter + 41 tagged exercises + 5 validated scenarios

### 🚧 In Progress

- Streamlit Phase 2.3 — image upload (Vision Agent) in chat
- Streamlit Phase 2.4 — chat polish + clear history + error states
- Biomechanical safety v1.0 — apply senior-review fixes (torsional loading, spinal shear, `JointAction` refinement, segment-coverage refusal)
- Backend `?debug=1` hook for agent trace visualization

### 🔮 Planned

- Streamlit Phases 3–5 — direct API explorers + progress visualizer + debug instrumentation
- Library-wide biomechanical tagging (2,800+ exercises) with physio audit
- LLM-as-judge evaluation harness for safety regression testing
- React Native mobile UI
- PostgresSaver checkpointer for multi-host LangGraph scaling
- Alembic migrations replacing `Base.metadata.create_all`

---

## 🔍 Inside the Engineering Decisions

A few non-obvious choices worth knowing about if you're reading the code:

- **ChromaDB single-threaded executor**: ChromaDB's C++ extension can segfault under Python's concurrent access on macOS. The `ChromaDBManager` singleton funnels all queries through a `ThreadPoolExecutor(max_workers=1)` to avoid this.
- **Lazy CLIP loading at startup**: CLIP weights are pre-loaded in the FastAPI lifespan to avoid mid-request segfaults under uvicorn worker reloads.
- **TDEE math lives in exactly one place**: `_compute_tdee()` in `app/agents/base.py` is the single source of truth (Mifflin-St Jeor). All three endpoints route through `_build_calorie_targets()` in `app/modules/ai/service.py` to derive their calorie targets.
- **The synthesis prompt is the master coach**: every multi-agent response goes through a final LLM pass that weaves specialist outputs into a single coaching message. Advisory vision queries bypass synthesis to avoid the LLM rewriting the analysis.
- **Auth dependency verifies the user still exists**: `get_current_user` in `app/core/security.py` doesn't just decode the JWT — it queries the DB to confirm the user is still present and active. Stolen tokens for deleted accounts fail immediately.

---

## 🤝 Contributing

This is currently a personal portfolio + research project. Architectural critiques and PRs against [`app/safety/IMPLEMENTATION_PLAN.md`](app/safety/IMPLEMENTATION_PLAN.md) (the production-rollout plan for biomechanical safety) are especially welcome.

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Built with ❤️ and a LOT of LangGraph nodes.**

</div>
