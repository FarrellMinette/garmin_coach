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
st.sidebar.header("Navigation")
view_mode = st.sidebar.radio("Select View", ["Metrics Overview", "3-Month Heatmap", "AI Coach Analysis"])

if view_mode == "Metrics Overview":
    st.subheader("Recent Training Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Activities Logged", len(df))
    col2.metric("Total Distance", f"{df['Distance_km'].sum():.1f} km")
    col3.metric("Avg Training Heart Rate", f"{df['Avg_HR'].mean():.1f} bpm")
    
    # --- Safe Data Cleaning for Plotly Scatter ---
    df_plot = df.copy()

    # Fill NaN heart rate or distance with sensible defaults
    df_plot['Avg_HR'] = df_plot['Avg_HR'].fillna(0)
    df_plot['Distance_km'] = df_plot['Distance_km'].fillna(0.1)

    # Ensure size is strictly positive (> 0) so Plotly can render the bubbles properly
    df_plot['Bubble_Size'] = df_plot['Distance_km'].apply(lambda x: max(float(x), 1.0))

    # Interactive Plotly Chart
    fig = px.scatter(
        df_plot, 
        x='Date', 
        y='Avg_HR', 
        color='Category', 
        size='Bubble_Size',
        hover_data=['Activity Type', 'Distance_km', 'Calories_kcal'], 
        title="Activity Heart Rate & Volume Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "3-Month Heatmap":
    st.subheader("3-Month Activity Consistency Matrix")
    
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
    
    colors = {'Running': '#e74c3c', 'Cycling': '#3498db', 'Strength': '#9b59b6', 
              'Walking': '#2ecc71', 'Water Sports': '#1abc9c', 'Racket Sports': '#f39c12', 'Other Activity': '#95a5a6'}
    
    fig_hm, ax = plt.subplots(figsize=(14, 6), dpi=300)
    for idx, row in cal_df.iterrows():
        wk_idx, y_pos = row['WeekIdx'], row['Visual_y']
        d = row['DateObj']
        ax.add_patch(patches.Circle((wk_idx, y_pos), 0.4, facecolor='#f8f9fa', edgecolor='#e9ecef', linewidth=0.5, zorder=1))
        if d in daily_activities:
            acts = daily_activities[d]
            if len(acts) == 1:
                ax.add_patch(patches.Circle((wk_idx, y_pos), 0.38, facecolor=colors.get(acts[0], '#333333'), edgecolor='none', zorder=2))
            elif len(acts) >= 2:
                ax.add_patch(patches.Wedge((wk_idx, y_pos), 0.38, 90, 270, facecolor=colors.get(acts[0], '#333333'), edgecolor='none', zorder=2))
                ax.add_patch(patches.Wedge((wk_idx, y_pos), 0.38, 270, 90, facecolor=colors.get(acts[1], '#e74c3c'), edgecolor='none', zorder=2))

    ax.set_xlim(-0.8, cal_df['WeekIdx'].max() + 0.8)
    ax.set_ylim(-1.0, 7.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, label in enumerate(day_labels):
        ax.text(-1.5, 6 - i, label, ha='right', va='center', fontsize=11, fontweight='bold', color='#495057')
        
    st.pyplot(fig_hm)

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
                        model='gemini-1.5-flash',
                        contents=prompt,
                    )
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"API Error: {e}")