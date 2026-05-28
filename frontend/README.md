# Frontend — AI Fitness Gym Dev Visualization

Streamlit-based developer visualization for the multi-agent AI Fitness Gym backend.
This is a **developer tool**, not the consumer-facing mobile app — it exposes agent
traces, RAG behavior, and raw API responses behind a 🐛 Developer mode toggle.

## Prerequisites

The FastAPI backend in `../app/` must be running, with MySQL, Redis, and ChromaDB up.

```bash
# from repo root, in one terminal:
uvicorn app.main:app --reload
```

## Setup

```bash
cd frontend
pip install -r requirements.txt
```

The app defaults to `http://localhost:8000` for the backend. To override:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit API_BASE_URL
```

## Run

```bash
streamlit run streamlit_app.py
```

App opens at `http://localhost:8501`.

## Phase 0 — what's here today

- **Login / Register** wired to `/api/auth/login` and `/api/auth/register`
- **Session-state auth** — JWT stored in `st.session_state.token` and
  injected as `Authorization: Bearer ...` on every backend call by
  `lib/api_client.py`. Token is lost on logout or tab close (no persistence).
- **Sidebar 🐛 Developer mode toggle** — functional now; per-agent trace
  rendering lands in Phase 5
- **Logout button**
- **Error normalization** — backend errors (4xx, 5xx, network) all surface
  through a single `ApiError` exception with a clean user-facing message;
  the SQLAlchemy / stack details never reach the UI.

## Coming next

| Phase | Adds |
|---|---|
| 1 | Profile onboarding + edit; Account page (feedback history, delete account) |
| 2 | Chat with multi-turn + image upload (Vision Agent); workout/meal/GIF renderers |
| 3 | Direct API pages: Generate Workout, Generate Diet, Domain Q&A |
| 4 | Progress Agent visualizer; thumbs-up/down feedback after every AI response |
| 5 | Backend `?debug=1` hook + per-agent trace tabs (intents, specialists, RAG, vision tier, timing) |
| 6 | Dockerfile + docker-compose entry; error states polish |

## File map

```
frontend/
├── streamlit_app.py              # entry: auth gate + welcome
├── lib/
│   ├── api_client.py             # requests wrapper, JWT injection, ApiError
│   ├── auth.py                   # login / register / logout / require_auth
│   └── debug.py                  # debug toggle + JSON panel helper
├── .streamlit/
│   ├── config.toml               # server port + telemetry off
│   └── secrets.toml.example      # API_BASE_URL template
├── requirements.txt
└── README.md
```
