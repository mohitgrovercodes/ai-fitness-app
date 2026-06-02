# Frontend — AI Fitness Gym Dev UI Client

A Streamlit-based developer panel and visualization tool for the multi-agent AI Fitness Gym backend. This tool enables developers and researchers to interact with the backend API, review agent execution traces, upload media, and verify biomechanical safety filters.

---

## 1. Prerequisites & Installation

The FastAPI backend must be running alongside MySQL, Redis, and ChromaDB.

```bash
# Install dependencies
cd frontend
pip install -r requirements.txt

# Start the Streamlit application
streamlit run streamlit_app.py
```
By default, the frontend connects to the backend at `http://localhost:8000`. To override this, configure `.streamlit/secrets.toml`.

---

## 2. Core Pages & Features

The interface consists of six core functional modules:

### 👤 Profile & Onboarding (`6_Profile.py`)
* Automatically detects user state. If a profile doesn't exist, it displays an onboarding questionnaire.
* Captures gender, age, height, weight, activity levels, dietary preferences, and explicit physical injuries.
* Saves onboarding state directly to the MySQL database via `POST /api/profile/onboarding`.

### 💬 Multi-Turn Chat & Vision (`3_Chat.py`)
* Supports continuous conversations with full session history.
* **VLM Upload Widget**: Includes an upload sidebar for image formats (`.png`, `.jpg`, `.jpeg`, `.webp`).
* **Intelligent Request Routing**:
  * If a food image is present, the page bundles the prompt and sends a `multipart/form-data` request to `POST /api/ai/chat-vision` to evaluate caloric density and macro splits.
  * If no image is present, it routes a standard JSON query to `POST /api/ai/chat`.

### 🏋️ Workout Plan Generator (`4_Workout.py`)
* Provides a customized parameter form pre-filled with the user's cached biometrics.
* Connects to `POST /api/ai/generate-workout` to generate safe training routines.
* Renders workout timelines as styled cards with targeting splits, sets/reps progression, coaching notes, and embedded video/GIF loops for correct form.

### 🥗 Diet Program Builder (`5_Diet.py`)
* Connects to `POST /api/ai/generate-diet` to calculate calorie target bands using Mifflin-St Jeor equations.
* Renders daily macronutrient strips (Proteins, Fats, Carbs) and color-coded meal cards detailing food quantities and allergen exclusions.

### 📚 Textbook Research Q&A (`8_Domain.py`)
* Submits queries to the vector store textbook database via `POST /api/ai/ask-domain`.
* Renders academic summaries alongside detailed citation footnotes linked back to validated sports-science texts.

### ⚙️ Account Management (`7_Account.py`)
* Displays overall user feedback metrics (thumbs-up/thumbs-down history).
* Contains a security-gated "Danger Zone" that allows accounts to be deleted only after entering the password for re-authentication.

---

## 3. Developer & Diagnostic Tools

Exposing system logs and LLM reasoning steps is critical for debugging multi-agent graphs.

* **Sidebar Developer Toggle 🐛**: Activating Developer Mode displays raw API response payloads, LangGraph execution paths, and intermediate prompt audits below the generated UI cards.
* **Error Normalization**: Stack traces and SQL connection exceptions are caught on the backend and translated into user-friendly `ApiError` alerts in the Streamlit client.
* **GIF Resynchronization**: When exercises are dynamically swapped or modified due to injury safety violations, the client automatically requests media verification, using fuzzy matching to display verified local files.
