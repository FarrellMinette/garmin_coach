# 🏃‍♂️ Garmin AI Training Dashboard & Coach

A fully functional, $0-cost custom Garmin web app and dashboard built with Python, Streamlit, and Google Gemini. It transforms raw Garmin Connect activity data into interactive calendar logs, per-sport trend analysis, and physiological AI training audits.

---

## 🌟 Features Currently Implemented

### 1. 🚀 Training & Consistency Hub

* **Core Metrics:** High-level overview displaying total logged activities, cumulative distance, and average training heart rate.
* **Unified Interactive Calendar Logs:** Tabbed interface for **Daily**, **Weekly** (1-row x 7-column grid spanning Monday through Sunday), and **Monthly** calendar grid views.
* **Rich Activity Metadata:** Clickable "More Info" popovers providing distance, average/max heart rate, calories, duration, pace, elevation, and recovery estimates.
* **3-Month Consistency Heatmap:** Visual breakdown of workout density over a rolling 90-day period with custom sport-coded colors and split activity indicators.

### 2. 📅 Interactive Log & Selector Controls

* **Year & Month Button Selectors:** Clean, horizontal button rows to quickly switch between years and active months without clutter.
* **Weekly 7-Day Grid:** Side-by-side view featuring custom metric cards slotted directly into each day column.
* **Smart Metric Handling:** Displays distance (km) for endurance sports (running, cycling, walking) and non-distance indicators (calories/strength) for gym sessions.
* **Clean Activity Standardizing:** Removes underscores and formats raw Garmin activity keys into properly capitalized titles (e.g., `indoor_cycling` ➔ `Indoor Cycling`).

### 3. 🏆 Per-Sport Insights & Trends

* Deterministic deep dives into specific activity categories (Running, Cycling, Strength, Walking, Water Sports, Racket Sports).
* Interactive progression and heart rate effort scaling charts using Plotly.
* Complete tabbed activity log with customizable columns.

### 4. 🤖 Gemini AI Physiological Training Audit

* Direct integration with `gemini-2.0-flash` via the Google GenAI SDK (`google-genai`).
* Generates evidence-based audits covering training intensity distribution (Zone 2 vs. high HR), cardiovascular strain, and targeted weekly workout recommendations.

---

## 🛠️ Tech Stack & Architecture

* **Frontend & Framework:** Streamlit (Streamlit Community Cloud)
* **Data Processing & Analytics:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `plotly`
* **Garmin API Ingestion:** `garminconnect`
* **AI Engine:** Google GenAI SDK (`google-genai` / `gemini-2.0-flash`)
* **Hosting:** Streamlit Cloud with `.streamlit/secrets.toml` secret management

---

## 🎯 Next Improvements & Future Roadmap

* [ ] **Modular Code Refactoring:** Break down `app.py` into a structured, multi-file codebase (`views/`, `utils.py`, `components/`) for improved maintainability.
* [ ] **Local Session Caching & Persistent Tokens:** Implement local session token storage for `garminconnect` to prevent re-authentication overhead on app reloads.
* [ ] **Advanced Heart Rate Zone Distribution:** Add dedicated Zone 1–Zone 5 time-in-zone charts per activity and per week.
* [ ] **Custom Workout & Target Logging:** Build a user input module to log subjective fatigue, sleep scores, and future weekly mileage targets.
* [ ] **Expanded AI Prompt Customization:** Allow toggling AI audit focus areas (e.g., marathon training prep, injury prevention, recovery strategy).

---

## 🚀 Getting Started Locally

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/garmin-ai-dashboard.git
cd garmin-ai-dashboard

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Configure Secrets:**
Create `.streamlit/secrets.toml` in the project root:
```toml
GARMIN_EMAIL = "your_garmin_email@example.com"
GARMIN_PASSWORD = "your_garmin_password"
GEMINI_API_KEY = "your_google_gemini_api_key"

```


4. **Run the Streamlit app:**
```bash
streamlit run app.py

```