import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
from garminconnect import Garmin
from google import genai

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Garmin AI Coach",
    page_icon="🏃‍♂️",
    layout="wide"
)

st.title("Garmin AI Training Dashboard")
st.markdown("Your personal, free, AI-powered endurance and biomechanics coach.")

# --- 2. AUTHENTICATION & DATA INGESTION ---
@st.cache_resource(ttl=3600)
def init_garmin_client(email, password):
    """Initializes and authenticates the Garmin client session."""
    try:
        api = Garmin(email, password)
        api.login()
        return api
    except Exception as e:
        st.error(f"Garmin Authentication Failed: {e}")
        return None

@st.cache_data(ttl=1800)
def fetch_garmin_activities(_api_client, limit=100):
    """Fetches recent activities from Garmin Connect."""
    try:
        activities = _api_client.get_activities(0, limit)
        return pd.DataFrame(activities)
    except Exception as e:
        st.warning(f"Could not fetch live activities: {e}. Falling back to local CSV if available.")
        if os.path.exists('Activities.csv'):
            return pd.read_csv('Activities.csv')
        return pd.DataFrame()

# Load secrets from Streamlit secrets management
try:
    G_EMAIL = st.secrets["GARMIN_EMAIL"]
    G_PASS = st.secrets["GARMIN_PASSWORD"]
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.info("Please configure your secrets in `.streamlit/secrets.toml` or Streamlit Cloud settings.")
    st.stop()

# Initialize API and load data
client = init_garmin_client(G_EMAIL, G_PASS)
if client:
    df = fetch_garmin_activities(client, limit=150)
else:
    if os.path.exists('Activities.csv'):
        df = pd.read_csv('Activities.csv')
    else:
        st.error("No data source available.")
        st.stop()

# --- 3. DATA CLEANING & TRANSFORMATION ---
if 'startTimeLocal' in df.columns:
    df['Date'] = pd.to_datetime(df['startTimeLocal'])
elif 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Standardize Activity Type column
if 'activityType' in df.columns and 'Activity Type' not in df.columns:
    df['Activity Type'] = df['activityType'].apply(lambda x: x.get('typeKey') if isinstance(x, dict) else str(x))

def categorize_activity(act):
    act = str(act).lower()
    if 'run' in act: return 'Running'
    elif 'cycle' in act or 'cycling' in act: return 'Cycling'
    elif 'strength' in act: return 'Strength'
    elif 'walk' in act: return 'Walking'
    elif 'swim' in act or 'snorkeling' in act: return 'Water Sports'
    elif 'squash' in act or 'tennis' in act or 'table tennis' in act: return 'Racket Sports'
    else: return 'Other Activity'

df['Category'] = df['Activity Type'].apply(categorize_activity)

# Extract numeric metrics safely
def safe_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace('--', 'NaN').str.replace(',', '').str.replace("'", ""), errors='coerce')

df['Distance_km'] = safe_numeric(df.get('distance', df.get('Distance', 0))) / 1000.0 # Garmin API usually returns meters
df['Calories_kcal'] = safe_numeric(df.get('calories', df.get('Calories', 0)))
df['Avg_HR'] = safe_numeric(df.get('averageHR', df.get('Avg HR', 0)))

# --- 4. DASHBOARD UI COMPONENTS ---
# --- 4. DASHBOARD UI NAVIGATION ---
st.sidebar.header("Navigation")
view_mode = st.sidebar.radio(
    "Select View", 
    [
        "Training Dashboard (Overview & Heatmap)", 
        "Weekly Calendar Log", 
        "Monthly Calendar Grid", 
        "Per-Sport Insights", 
        "AI Coach Analysis"
    ]
)

sport_colors = {
    'Running': '#e74c3c', 'Cycling': '#3498db', 'Strength': '#9b59b6', 
    'Walking': '#2ecc71', 'Water Sports': '#1abc9c', 'Racket Sports': '#f39c12', 'Other Activity': '#95a5a6'
}
sport_logos = {
    'Running': '🏃‍♂️', 'Cycling': '🚴‍♂️', 'Strength': '🏋️‍♂️', 'Walking': '🚶‍♂️', 
    'Water Sports': '🏊‍♂️', 'Racket Sports': '🎾', 'Other Activity': '🏅'
}

# --- VIEW 1: UNIFIED TRAINING DASHBOARD ---
if view_mode == "Training Dashboard (Overview & Heatmap)":
    st.subheader("🚀 Training & Consistency Hub")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Activities Logged", len(df))
    col2.metric("Total Distance", f"{df['Distance_km'].sum():.1f} km")
    col3.metric("Avg Training Heart Rate", f"{df['Avg_HR'].mean():.1f} bpm")
    
    st.markdown("---")
    
    # 3-Month Consistency Heatmap styled with dark background matching Plotly charts
    st.markdown("### 🗓️ 3-Month Consistency Heatmap")
    end_date = df['Date'].max() if not df.empty else datetime.now()
    start_date = end_date - timedelta(days=90)
    df_recent = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    df_recent['Day'] = df_recent['Date'].dt.date
    
    daily_activities = df_recent.groupby('Day')['Category'].apply(list).to_dict()
    date_range = pd.date_range(start=start_date, end=end_date)
    cal_df = pd.DataFrame({'Date': date_range})
    cal_df['DateObj'] = cal_df['Date'].dt.date
    cal_df['Week'] = cal_df['Date'].dt.isocalendar().week
    
    unique_weeks = cal_df['Week'].unique()
    week_map = {w: i for i, w in enumerate(unique_weeks)}
    cal_df['WeekIdx'] = cal_df['Week'].map(week_map)
    cal_df['Visual_y'] = 6 - cal_df['Date'].dt.dayofweek
    
    fig_hm, ax = plt.subplots(figsize=(14, 4.5), dpi=300)
    fig_hm.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    for idx, row in cal_df.iterrows():
        wk_idx, y_pos = row['WeekIdx'], row['Visual_y']
        d = row['DateObj']
        ax.add_patch(patches.Circle((wk_idx, y_pos), 0.4, facecolor='#262730', edgecolor='#363842', linewidth=0.5, zorder=1))
        if d in daily_activities:
            acts = daily_activities[d]
            if len(acts) == 1:
                ax.add_patch(patches.Circle((wk_idx, y_pos), 0.38, facecolor=sport_colors.get(acts[0], '#333333'), edgecolor='none', zorder=2))
            elif len(acts) >= 2:
                ax.add_patch(patches.Wedge((wk_idx, y_pos), 0.38, 90, 270, facecolor=sport_colors.get(acts[0], '#333333'), edgecolor='none', zorder=2))
                ax.add_patch(patches.Wedge((wk_idx, y_pos), 0.38, 270, 90, facecolor=sport_colors.get(acts[1], '#e74c3c'), edgecolor='none', zorder=2))

    ax.set_xlim(-0.8, cal_df['WeekIdx'].max() + 0.8)
    ax.set_ylim(-1.0, 7.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, label in enumerate(day_labels):
        ax.text(-1.5, 6 - i, label, ha='right', va='center', fontsize=10, fontweight='bold', color='#fafafa')
        
    # Add Legend to Heatmap
    legend_elements = [patches.Patch(facecolor=color, label=sport) for sport, color in sport_colors.items()]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, facecolor='#262730', edgecolor='none', labelcolor='white')
    
    st.pyplot(fig_hm)

# --- VIEW 2: WEEKLY CALENDAR LOG ---
elif view_mode == "Weekly Calendar Log":
    st.subheader("📅 Weekly Calendar View (Last 7 Days)")
    
    if df.empty:
        st.warning("No data available.")
    else:
        latest_date = df['Date'].max()
        week_start = latest_date - timedelta(days=6)
        week_days = pd.date_range(start=week_start, end=latest_date)
        
        df['DateOnly'] = df['Date'].dt.date
        
        for single_date in week_days:
            d_obj = single_date.date()
            day_name = single_date.strftime('%A, %b %d, %Y')
            day_acts = df[df['DateOnly'] == d_obj]
            
            with st.container():
                if not day_acts.empty:
                    st.markdown(f"#### 🟢 {day_name} (Active)")
                    for _, row in day_acts.iterrows():
                        cat = row.get('Category', 'Other Activity')
                        logo = sport_logos.get(cat, '🏅')
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.markdown(f"**{logo} {row.get('ActivityType', row.get('Activity Type', cat))}**")
                        col2.metric("Distance", f"{row.get('Distance_km', 0):.2f} km")
                        col3.metric("Avg HR", f"{row.get('Avg_HR', 0)} bpm")
                        col4.metric("Calories", f"{row.get('Calories_kcal', 0)} kcal")
                        col5.metric("Training Load", cat)
                else:
                    st.markdown(f"#### ⚪ {day_name} — *Rest Day / No Activity*")
                st.markdown("---")

# --- VIEW 3: MONTHLY CALENDAR GRID ---
elif view_mode == "Monthly Calendar Grid":
    st.subheader("🗓️ Monthly Calendar Grid & Activity Slots")
    
    if df.empty:
        st.warning("No data available.")
    else:
        df['Month_Year'] = df['Date'].dt.to_period('M')
        available_months = sorted(df['Month_Year'].unique(), reverse=True)
        selected_month = st.selectbox("Select Month", available_months, format_func=lambda x: x.strftime('%B %Y'))
        
        m_df = df[df['Month_Year'] == selected_month].copy()
        m_df['DayNum'] = m_df['Date'].dt.day
        daily_map = m_df.groupby('DayNum').apply(lambda x: x.to_dict('records')).to_dict()
        
        year = selected_month.year
        month = selected_month.month
        import calendar
        cal_matrix = calendar.monthcalendar(year, month)
        
        st.markdown(f"### {selected_month.strftime('%B %Y')} Calendar")
        cols = st.columns(7)
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, day_name in enumerate(weekdays):
            cols[i].markdown(f"**{day_name}**")
            
        for week in cal_matrix:
            cols = st.columns(7)
            for i, day in enumerate(week):
                if day == 0:
                    cols[i].markdown("")
                else:
                    with cols[i]:
                        st.markdown(f"**{day}**")
                        if day in daily_map:
                            for act in daily_map[day]:
                                cat = act.get('Category', 'Other Activity')
                                logo = sport_logos.get(cat, '🏅')
                                dist = act.get('Distance_km', 0)
                                hr = act.get('Avg_HR', 0)
                                st.markdown(f"<small>{logo} **{dist:.1f}km**<br>HR: {hr}bpm</small>", unsafe_allow_html=True)
                        else:
                            st.markdown("<small style='color:grey;'>Rest</small>", unsafe_allow_html=True)
            st.markdown("---")

# --- VIEW 4: PER-SPORT INSIGHTS ---
elif view_mode == "Per-Sport Insights":
    st.subheader("🏆 Per-Sport Deterministic Insights")
    
    if df.empty:
        st.warning("No data available.")
    else:
        sports = df['Category'].unique()
        selected_sport = st.selectbox("Select Sport Category", sports)
        
        sport_df = df[df['Category'] == selected_sport]
        logo = sport_logos.get(selected_sport, '🏅')
        
        st.markdown(f"### {logo} Analytics for: {selected_sport}")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total Sessions", len(sport_df))
        sc2.metric("Total Distance", f"{sport_df['Distance_km'].sum():.1f} km")
        sc3.metric("Average Session HR", f"{sport_df['Avg_HR'].mean():.1f} bpm")
        
        st.markdown("#### Key Per-Sport Observations:")
        max_dist_row = sport_df.loc[sport_df['Distance_km'].idxmax()] if not sport_df.empty else None
        if max_dist_row is not None:
            st.info(f"🚀 **Longest {selected_sport} Session:** {max_dist_row['Distance_km']:.2f} km on {pd.to_datetime(max_dist_row['Date']).strftime('%b %d, %Y')} with an average HR of {max_dist_row['Avg_HR']} bpm.")
            
        max_hr_row = sport_df.loc[sport_df['Avg_HR'].idxmax()] if not sport_df.empty else None
        if max_hr_row is not None:
            st.success(f"🔥 **Highest Intensity {selected_sport} Session:** Averaged {max_hr_row['Avg_HR']} bpm on {pd.to_datetime(max_hr_row['Date']).strftime('%b %d, %Y')}.")

# --- VIEW 5: AI COACH ANALYSIS ---
elif view_mode == "AI Coach Analysis":
    st.subheader("Gemini AI Training Audit")
    if st.button("Generate Physiological Audit"):
        if df.empty:
            st.warning("No activity data available to analyze.")
        else:
            with st.spinner("Analyzing your Garmin training load and heart rate zones..."):
                client_ai = genai.Client(api_key=GEMINI_KEY)
                summary_stats = df[['Activity Type', 'Distance_km', 'Calories_kcal', 'Avg_HR']].head(20).to_string()
                prompt = f"""
                You are a rigorous sports scientist and exercise physiologist. Review these recent Garmin activities:
                {summary_stats}
                
                Provide a precise, scientifically accurate audit of:
                1. Training intensity distribution (polarization / high HR vs Zone 2 balance).
                2. Cardiovascular strain and recovery implications.
                3. Two actionable, specific workouts to improve efficiency this week.
                Keep it structured, punchy, and evidence-based.
                """
                
                try:
                    response = client_ai.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=prompt,
                    )
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"API Error: {e}")