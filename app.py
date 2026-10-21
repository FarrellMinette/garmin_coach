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
view_mode = st.sidebar.radio("Select View", ["Metrics Overview", "3-Month Heatmap", "Periodic Summaries", "AI Coach Analysis"])

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

elif view_mode == "Periodic Summaries":
    st.subheader("Periodic Training Aggregates")
    
    if df.empty:
        st.warning("No data available.")
    else:
        # Ensure Date is set as datetime index for resampling/period grouping
        df_per = df.copy()
        df_per['Date'] = pd.to_datetime(df_per['Date'])
        
        # Create period columns
        df_per['Week_Period'] = df_per['Date'].dt.to_period('W').dt.start_time.dt.strftime('%Y-%m-%d')
        df_per['Month_Period'] = df_per['Date'].dt.to_period('M').dt.strftime('%B %Y')
        df_per['Quarter_Period'] = df_per['Date'].dt.to_period('Q').astype(str)
        
        period_tab = st.selectbox("Select Time Horizon", ["Weekly Summary", "Monthly Summary", "3-Month / Quarterly Summary"])
        
        if period_tab == "Weekly Summary":
            st.markdown("### 📅 Weekly Breakdown")
            group_col = 'Week_Period'
            sort_order = False # Most recent first
        elif period_tab == "Monthly Summary":
            st.markdown("### 🗓️ Monthly Breakdown")
            group_col = 'Month_Period'
            sort_order = False
        else:
            st.markdown("### 📊 Quarterly (3-Month) Breakdown")
            group_col = 'Quarter_Period'
            sort_order = False

        # Aggregate data safely
        summary_df = df_per.groupby(group_col).agg(
            Total_Activities=('Category', 'count'),
            Total_Distance_km=('Distance_km', 'sum'),
            Total_Calories=('Calories_kcal', 'sum'),
            Avg_Heart_Rate=('Avg_HR', 'mean'),
            Primary_Activity=('Category', lambda x: x.mode()[0] if not x.empty else 'Mixed')
        ).reset_index()
        
        # Format numeric outputs for clean presentation
        summary_df['Total_Distance_km'] = summary_df['Total_Distance_km'].round(2)
        summary_df['Avg_Heart_Rate'] = summary_df['Avg_Heart_Rate'].round(1)
        summary_df['Total_Calories'] = summary_df['Total_Calories'].round(0)
        
        summary_df = summary_df.sort_values(by=group_col, ascending=sort_order)
        
        # Display as a clean interactive dataframe table
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Quick Static Metric Insights Cards based on the selection
        st.markdown("---")
        st.markdown("### 💡 Quick Deterministic Insights")
        
        max_row = summary_df.loc[summary_df['Total_Distance_km'].idxmax()] if not summary_df.empty else None
        if max_row is not None:
            st.info(f"🏆 **Peak Period:** Your highest distance period was **{max_row[group_col]}** with a total of **{max_row['Total_Distance_km']} km** across **{max_row['Total_Activities']} activities**.")
            
        avg_dist_period = summary_df['Total_Distance_km'].mean()
        st.success(f"📈 **Consistency Baseline:** You average **{avg_dist_period:.1f} km** per {period_tab.split()[0].lower()} tracked.")

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
                        model='gemini-3.1-flash-lite',
                        contents=prompt,
                    )
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"API Error: {e}")