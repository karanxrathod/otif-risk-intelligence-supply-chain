import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & UNDERSTANDING
# ==========================================
def load_and_merge_data():
    print("\n[1/7] Loading Data...")
    try:
        # Load Dimensions
        dim_cust = pd.read_csv("database/dim_customers.csv")
        dim_prod = pd.read_csv("database/dim_products.csv")
        dim_date = pd.read_csv("database/dim_date.csv")
        dim_targets = pd.read_csv("database/dim_targets_orders.csv")
        
        # Load Facts
        # Note: In AtliQ Mart data, order_lines contains line-level details
        fact_lines = pd.read_csv("database/fact_order_lines.csv")
        fact_agg = pd.read_csv("database/fact_orders_aggregate.csv")
        
        print(f"   -> Loaded {len(fact_lines)} order lines.")
        
        # Merge Information
        # Convert Dates (assuming format is suitable, otherwise explicit format needed)
        # Checking schema first
        # fact_lines probably has 'order_id', 'customer_id', 'product_id', 'order_placement_date', 'delivery_date', 'agreed_delivery_date'
        
        # Merge Products to get Category
        df = fact_lines.merge(dim_prod, on="product_id", how="left")
        
        # Merge Customers to get Channel/City
        df = df.merge(dim_cust, on="customer_id", how="left")

        # Merge Aggregate to get Ground Truth OTIF Labels
        # fact_agg has 'order_id', 'on_time', 'in_full'
        if 'on_time' in fact_agg.columns:
            df = df.merge(fact_agg[['order_id', 'on_time', 'in_full']], on='order_id', how='left')
        
        # Ensure Date columns are datetime
        date_cols = ['agreed_delivery_date', 'actual_delivery_date', 'order_placement_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce') # Handle potential parsing errors
        
        print(f"   -> Merged Dataset Shape: {df.shape}")
        return df, dim_targets
        
    except FileNotFoundError as e:
        print(f"   [ERROR] File not found: {e}")
        return None, None

# ==========================================
# 2. FEATURE ENGINEERING (OTIF METRICS)
# ==========================================
def calculate_otif_features(df):
    print("\n[2/7] Calculating OTIF Metrics...")
    
    # 1. Use Ground Truth dataset columns if available
    if 'on_time' in df.columns and 'in_full' in df.columns:
        print("   -> Using Ground Truth 'on_time' and 'in_full' from dataset.")
        df['is_on_time'] = df['on_time']
        df['is_in_full'] = df['in_full']
    else:
        # Fallback to calculation
        print("   -> Calculating OTIF from dates (Fallback).")
        df['is_on_time'] = np.where(
            (df['actual_delivery_date'].notna()) & (df['actual_delivery_date'] <= df['agreed_delivery_date']), 
            1, 0
        )
        if 'order_qty' in df.columns and 'delivery_qty' in df.columns:
            df['is_in_full'] = np.where(df['delivery_qty'] >= df['order_qty'], 1, 0)
        else:
            df['is_in_full'] = 0

    # OTIF: Must be BOTH
    df['is_otif'] = df['is_on_time'] * df['is_in_full']
    
    # --- Feature Engineering for ML ---
    # 1. Lead Time (Agreed Date - Placement Date)
    if 'agreed_delivery_date' in df.columns and 'order_placement_date' in df.columns:
         df['lead_time_planned'] = (df['agreed_delivery_date'] - df['order_placement_date']).dt.days.fillna(0)
    else:
         df['lead_time_planned'] = 0
    
    # 2. Day of Week / Month
    df['order_day'] = df['order_placement_date'].dt.dayofweek
    df['order_month'] = df['order_placement_date'].dt.month
    
    # 3. Delay (Target Variable helper)
    if 'actual_delivery_date' in df.columns:
        df['delay_days'] = (df['actual_delivery_date'] - df['agreed_delivery_date']).dt.days.fillna(0)
    
    # 4. Target Label: OTIF Failure (1 = Fail, 0 = Success)
    df['risk_label'] = 1 - df['is_otif']
    
    print(f"   -> Average OTIF %: {df['is_otif'].mean():.2%}")
    
    return df

# ==========================================
# 3. DEMAND FORECASTING LAYER
# ==========================================
def generate_demand_features(df):
    print("\n[3/7] Generating Demand Signals...")
    
    # Sort by Product and Date
    df = df.sort_values(by=['product_id', 'order_placement_date'])
    
    # Rolling Demand (7-day window) per product
    # We need to aggregate orders per day first to get daily demand, then map back?
    # Or just rolling sum on the sorted lines (approximation).
    # Better: Group by Product, Date -> Sum Qty -> Rolling Mean -> Merge back
    
    daily_demand = df.groupby(['product_id', 'order_placement_date'])['order_qty'].sum().reset_index()
    daily_demand['rolling_7d_avg'] = daily_demand.groupby('product_id')['order_qty'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    daily_demand['demand_volatility'] = daily_demand.groupby('product_id')['order_qty'].transform(lambda x: x.rolling(window=7, min_periods=1).std()).fillna(0)
    
    # Merge back to main df
    df = df.merge(daily_demand[['product_id', 'order_placement_date', 'rolling_7d_avg', 'demand_volatility']], 
                  on=['product_id', 'order_placement_date'], how='left')
    
    return df

# ==========================================
# 4. RISK PREDICTION MODEL (ML)
# ==========================================
def train_risk_model(df):
    print("\n[4/7] Training OTIF Risk Model...")
    
    # Select Features
    # Numeric: Lead Time, Order Qty, Rolling Demand, Volatility
    # Categorical (Encoded): Customer, Product Category
    
    # Simple Encoding
    cat_cols = ['customer_id', 'category', 'city'] # Assuming these exist
    for col in cat_cols:
        if col in df.columns:
            df[col + '_enc'] = df[col].astype('category').cat.codes
            
    features = ['lead_time_planned', 'order_qty', 'rolling_7d_avg', 'demand_volatility', 'order_day']
    # Add available encoded cols
    for col in cat_cols:
        if col + '_enc' in df.columns:
            features.append(col + '_enc')
            
    # Drop rows with NaNs in features
    model_df = df.dropna(subset=features)
    
    X = model_df[features]
    y = model_df['risk_label']
    
    # Time-based Split (Train on past, Test on future)
    # Finding a split date (e.g., last 20% of data)
    dates = model_df['order_placement_date'].sort_values().unique()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    
    train_mask = model_df['order_placement_date'] < split_date
    test_mask = model_df['order_placement_date'] >= split_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(f"   -> Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"   -> Recall (Failure Detection): {recall_score(y_test, y_pred):.2f}")
    print(f"   -> ROC-AUC Score: {roc_auc_score(y_test, y_prob):.2f}")
    
    # Attach predictions to test set for simulation
    result_df = model_df.loc[test_mask].copy()
    result_df['pred_failure_prob'] = y_prob
    result_df['pred_failure_flag'] = y_pred
    
    return rf, result_df

# ==========================================
# 5. DECISION SIMULATION LAYER
# ==========================================
def simulate_interventions(result_df):
    print("\n[5/7] Simulating Proactive Decisions...")
    
    # Logic: 
    # If Predicted Failure Probability > 0.7 -> ACTION: "Expedite Shipping"
    # Assumption: Expediting reduces failure risk by 50% (simple heuristic)
    
    threshold = 0.7
    
    # Identify Risk Orders
    high_risk_orders = result_df[result_df['pred_failure_prob'] > threshold]
    print(f"   -> Detected {len(high_risk_orders)} High-Risk Orders (out of {len(result_df)})")
    
    # Calculate Impact
    # Original Failures (Ground Truth)
    original_failures = result_df['risk_label'].sum()
    
    # Simulated New Failures
    # If we intervene, we assume we convert a Risk=1 to Risk=0 with 80% success rate
    saved_orders = 0
    for idx, row in high_risk_orders.iterrows():
        # If it was actually going to fail (Ground Truth = 1)
        if row['risk_label'] == 1:
            # We intervene. Success chance?
            if np.random.random() > 0.2: # 80% success rate of intervention
                saved_orders += 1
                
    print("\n   [BUSINESS IMPACT ANALYSIS]")
    print(f"   Original OTIF Failures: {original_failures}")
    print(f"   Failures Prevented by AI: {saved_orders}")
    print(f"   New Simulated OTIF Failures: {original_failures - saved_orders}")
    
    return high_risk_orders

# ==========================================
# 6. REPORTING
# ==========================================
def generate_report(df, high_risk_orders):
    print("\n[6/7] Generating Logic Report...")
    print("-" * 40)
    print("      ATLIQ MART - SUPPLY CHAIN AI REPORT      ")
    print("-" * 40)
    
    # KPIs
    current_otif = df['is_otif'].mean()
    print(f"Global OTIF Score: {current_otif:.1%}")
    
    top_issues = df[df['is_otif'] == 0]['category'].value_counts().head(3)
    print(f"\nTop Categories Contributing to Failure:\n{top_issues.to_string()}")
    
    print("\n[Actionable Insight]")
    print(f"Recommend immediate review of {len(high_risk_orders)} orders flagged for potential delay.")
    
# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load
    df_raw, dim_targets = load_and_merge_data()
    
    if df_raw is not None:
        # 2. Features
        df_proc = calculate_otif_features(df_raw)
        
        # 3. Demand Signal
        df_proc = generate_demand_features(df_proc)
        
        # 4. Train Model
        model, test_results = train_risk_model(df_proc)
        
        # 5. Simulate
        high_risk = simulate_interventions(test_results)
        
        # 6. Report
        generate_report(df_proc, high_risk)
        
        print("\n[7/7] Pipeline Complete. Ready for Judge Evaluation. 🚀")
    else:
        print("Pipeline aborted due to data loading error.")
