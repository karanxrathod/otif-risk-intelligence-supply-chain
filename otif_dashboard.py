import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from otif_pipeline import load_and_merge_data, calculate_otif_features, generate_demand_features, train_risk_model, simulate_interventions

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AtliQ Mart | Intelligent Supply Chain",
    layout="wide",
    page_icon="🚚",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING (Aesthetics)
# ==========================================
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
    }
    
    /* Headers */
    h1 {
        color: #1E3A8A; /* Navy Blue */
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #334155;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING (Cached)
# ==========================================
@st.cache_data
def get_data_and_model():
    # 1. Load
    with st.spinner("Connecting to Data Warehouse..."):
        df_raw, dim_targets = load_and_merge_data()
        
    if df_raw is None:
        return None, None, None, None
        
    # 2. Process
    with st.spinner("Calculating OTIF Metrics & Demand Signals..."):
        df_proc = calculate_otif_features(df_raw)
        df_proc = generate_demand_features(df_proc)
    
    # 3. Train
    with st.spinner("Training Predictive Risk Models..."):
        model, test_results = train_risk_model(df_proc)
    
    return df_proc, model, test_results, dim_targets

# Load Data
df, model, test_results, dim_targets = get_data_and_model()

if df is None:
    st.error("⚠️ Critical Error: Unable to load dataset. Please verify 'database' folder exists locally.")
    st.stop()

# ==========================================
# SIDEBAR NAVIGATION & FILTERS
# ==========================================
from streamlit_option_menu import option_menu

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830312.png", width=50) # Placeholder Logo
    st.title("AtliQ Supply Chain")
    st.markdown("---")
    
    # Professional Menu
    menu_selection = option_menu(
        menu_title=None, # Hide title
        options=["Home", "Dashboard", "Demand & Risk", "Control Tower"],
        icons=["house", "speedometer2", "graph-up-arrow", "broadcast"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "black", "font-size": "18px"}, 
            "nav-link": {"color": "black", "font-size": "16px", "text-align": "left", "margin":"5px", "--hover-color": "#e0e0e0"},
            "nav-link-selected": {"background-color": "black", "color": "white", "font-weight": "bold"},
        }
    )
    
    st.markdown("---")
    st.subheader("🛠️ Global Filters")
    
    # 1. Category Filter
    all_categories = sorted(df['category'].unique().tolist()) if 'category' in df.columns else []
    selected_cats = st.multiselect("Category", all_categories, default=all_categories[:3])
    
    st.markdown("---")
    if st.button("🔄 Reload System", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
        
    st.caption(f"Ver: 2.1 | Orders: {len(df):,}")

# Apply Filters
if selected_cats:
    df_filtered = df[df['category'].isin(selected_cats)]
else:
    df_filtered = df

# ==========================================
# PAGE ROUTING
# ==========================================

# --- 1. HOME OVERVIEW ---
if menu_selection == "Home":
    st.title("Welcome to the Supply Chain Nerve Center")
    st.markdown("### *Predict anomalies, Prevent failures, Perfect delivery.*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **System Capabilities:**
        1.  **🔍 Real-Time Visibility:** Monitor global OTIF, On-Time, and In-Full percentages.
        2.  **🔮 Predictive Intelligence:** AI models analyze demand volatility to predict failures.
        3.  **🛡️ Proactive Control:** Run "What-If" simulations to reroute stock.
        
        **How to use:**
        - Go to **Dashboard** for high-level KPIs.
        - Use **Control Tower** to act on "High Risk" orders.
        """)
        
    with col2:
        # Mini High-Level Metric
        current_otif = df_filtered['is_otif'].mean()
        target = 0.65
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_otif * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Net OTIF %"},
            delta = {'reference': target * 100, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1E3A8A"},
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': target * 100}}))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

# --- 2. EXECUTIVE DASHBOARD ---
elif menu_selection == "Dashboard":
    st.title("📊 Executive Performance")
    st.caption(f"Showing data for: {', '.join(selected_cats) if selected_cats else 'All Categories'}")
    
    # Top Row: KPIs
    current_otif = df_filtered['is_otif'].mean()
    on_time = df_filtered['is_on_time'].mean()
    in_full = df_filtered['is_in_full'].mean()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 OTIF %", f"{current_otif:.1%}", delta="-37% vs Target" if current_otif < 0.65 else "On Track")
    c2.metric("⏰ On-Time %", f"{on_time:.1%}")
    c3.metric("🛒 In-Full %", f"{in_full:.1%}")
    c4.metric("📝 Total Orders", f"{len(df_filtered):,}")
    
    st.markdown("---")
    
    # Middle Row: Charts
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📉 OTIF Trend")
        # Daily OTIF Trend
        daily_trend = df_filtered.groupby('order_placement_date')[['is_otif', 'is_on_time', 'is_in_full']].mean().reset_index()
        fig_trend = px.line(daily_trend, x='order_placement_date', y=['is_otif', 'is_on_time', 'is_in_full'],
                            color_discrete_map={'is_otif': '#1E3A8A', 'is_on_time': '#10B981', 'is_in_full': '#F59E0B'})
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_right:
        st.subheader("⚠️ Top Failure Drivers")
        if 'city' in df_filtered.columns:
            city_otif = df_filtered.groupby('city')['is_otif'].mean().sort_values().head(5)
            st.bar_chart(city_otif, color="#EF4444")

# --- 3. DEMAND SENSING ---
elif menu_selection == "Demand & Risk":
    st.title("🧠 Demand Sensing & Risk AI")
    
    product_list = sorted(df_filtered['product_name'].unique()) if 'product_name' in df_filtered.columns else []
    if not product_list:
        st.warning("No products found in selected categories.")
    else:
        selected_prod = st.selectbox("Select Product:", product_list)
        prod_data = df[df['product_name' if 'product_name' in df.columns else 'product_id'] == selected_prod].sort_values('order_placement_date')
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader(f"Demand: {selected_prod}")
            fig_demand = px.bar(prod_data, x='order_placement_date', y='order_qty')
            fig_demand.add_scatter(x=prod_data['order_placement_date'], y=prod_data['rolling_7d_avg'], name='7-Day Avg')
            st.plotly_chart(fig_demand, use_container_width=True)
        with c2:
            st.metric("Risk Score", f"{(1 - prod_data['is_otif'].mean()):.1%}")

# --- 4. CONTROL TOWER ---
elif menu_selection == "Control Tower":
    st.title("🚨 Supply Chain Control Tower")
    st.caption("AI-Powered Exception Management")
    
    # Simulation Inputs
    with st.expander("🛠️ Simulation Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Risk Threshold (Trigger Alert)", 0.0, 1.0, 0.7, 0.05, help="Orders with predicted failure probability above this value will be flagged.")
        with col2:
            success_rate = st.slider("Intervention Success Probability", 0.0, 1.0, 0.85, help="Estimated success rate of rerouting or expediting.")

    # Get Test Results relevant to filters? 
    # Note: 'test_results' is global. Applying category filters to it:
    if selected_cats:
        test_data_filtered = test_results[test_results['category'].isin(selected_cats)]
    else:
        test_data_filtered = test_results
        
    # Identification
    high_risk_orders = test_data_filtered[test_data_filtered['pred_failure_prob'] > threshold]
    
    # Impact Calc
    original_failures = test_data_filtered['risk_label'].sum()
    potential_saves = int(len(high_risk_orders[high_risk_orders['risk_label'] == 1]) * success_rate)
    new_failures = original_failures - potential_saves
    
    if len(test_data_filtered) > 0:
        orig_otif = 1 - (original_failures / len(test_data_filtered))
        new_otif = 1 - (new_failures / len(test_data_filtered))
    else:
        orig_otif = 0
        new_otif = 0
    
    st.divider()
    
    # Results visualization
    st.subheader("simulation Results")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("🚨 High Risk Orders Detected", f"{len(high_risk_orders)}", help="Orders requiring immediate attention.")
    m2.metric("🛡️ Potential Saved Orders", f"{potential_saves}", help="Orders saved if action is taken.")
    m3.metric("📈 Projected OTIF Uplift", f"{new_otif:.1%}", delta=f"+{(new_otif - orig_otif):.1%}")
    
    st.success(f"**Insight:** By intervening on these **{len(high_risk_orders)}** orders, you can boost OTIF from **{orig_otif:.1%}** to **{new_otif:.1%}**.")
    
    st.subheader("📋 Action Queue (Priority List)")
    st.dataframe(
        high_risk_orders[['order_id', 'product_name', 'customer_name', 'agreed_delivery_date', 'pred_failure_prob']]
        .sort_values(by='pred_failure_prob', ascending=False)
        .style.format({'pred_failure_prob': '{:.1%}'})
        .background_gradient(subset=['pred_failure_prob'], cmap='Reds'),
        use_container_width=True
    )
    
    csv = high_risk_orders.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Action List (CSV)", csv, "high_risk_orders.csv", "text/csv")
