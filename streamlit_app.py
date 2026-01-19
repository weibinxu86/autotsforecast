import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from autotsforecast import AutoForecaster
from autotsforecast.models.base import LinearForecaster, MovingAverageForecaster, VARForecaster
from autotsforecast.models.external import (
    ARIMAForecaster, ETSForecaster, ProphetForecaster,
    XGBoostForecaster, RandomForestForecaster
)

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#0E1117'
plt.rcParams['axes.facecolor'] = '#262730'
plt.rcParams['axes.edgecolor'] = '#4A4A4A'
plt.rcParams['text.color'] = '#FAFAFA'
plt.rcParams['axes.labelcolor'] = '#FAFAFA'
plt.rcParams['xtick.color'] = '#FAFAFA'
plt.rcParams['ytick.color'] = '#FAFAFA'
plt.rcParams['grid.color'] = '#4A4A4A'


@dataclass
class AppState:
    data: pd.DataFrame
    date_col: Optional[str]
    target_cols: List[str]
    covariate_cols: List[str]
    horizon: int
    model_choice: str
    ma_window: int
    n_lags: int
    per_series_covariates: bool
    covariate_mapping: Dict[str, List[str]]


def generate_sample_data() -> pd.DataFrame:
    """Generate sample multi-series time series data with covariates"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    
    # Generate covariates
    temperature = np.random.uniform(60, 90, 180)
    advertising_spend = np.random.uniform(1000, 5000, 180)
    competitor_price = np.random.uniform(10, 30, 180)
    promotion = np.random.choice([0, 1], 180, p=[0.7, 0.3])
    
    # Generate more realistic data with trends and seasonality
    # Product A is influenced by temperature and advertising
    trend_a = np.linspace(1000, 1200, 180)
    seasonality_a = 50 * np.sin(np.linspace(0, 8 * np.pi, 180))
    temp_effect_a = (temperature - 75) * 2  # Higher temp = more sales
    ad_effect_a = advertising_spend * 0.01
    noise_a = np.random.randn(180) * 15
    
    # Product B is influenced by competitor price and promotions
    trend_b = np.linspace(500, 650, 180)
    seasonality_b = 30 * np.sin(np.linspace(0, 6 * np.pi, 180))
    price_effect_b = (25 - competitor_price) * 3  # Lower competitor price = less sales
    promo_effect_b = promotion * 50  # Promotions boost sales
    noise_b = np.random.randn(180) * 10
    
    data = pd.DataFrame({
        "sales_product_a": trend_a + seasonality_a + temp_effect_a * 0.3 + ad_effect_a * 0.2 + noise_a,
        "sales_product_b": trend_b + seasonality_b + price_effect_b * 0.2 + promo_effect_b + noise_b,
        "temperature": temperature,
        "advertising_spend": advertising_spend,
        "competitor_price": competitor_price,
        "promotion": promotion,
    }, index=dates)
    data.index.name = "date"
    return data.reset_index()


def infer_date_column(df: pd.DataFrame) -> Optional[str]:
    """Automatically detect date column"""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return None


def prepare_timeseries(df: pd.DataFrame, date_col: str, target_cols: List[str]) -> pd.DataFrame:
    """Convert dataframe to time series format"""
    ts = df.copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])
    ts = ts.sort_values(date_col).set_index(date_col)
    ts = ts[target_cols]
    ts = ts.apply(pd.to_numeric, errors="coerce")
    ts = ts.dropna(how="all")
    return ts


def calculate_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive forecast metrics"""
    metrics = {}
    for col in y_true.columns:
        if col not in y_pred.columns:
            continue
        true_vals = y_true[col].values
        pred_vals = y_pred[col].values
        
        # Remove NaN values
        mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
        true_vals = true_vals[mask]
        pred_vals = pred_vals[mask]
        
        if len(true_vals) == 0:
            continue
            
        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
        mae = np.mean(np.abs(true_vals - pred_vals))
        mape = np.mean(np.abs((true_vals - pred_vals) / (true_vals + 1e-10))) * 100
        
        # R-squared
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        metrics[col] = {
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "MAPE (%)": round(mape, 2),
            "R²": round(r2, 3)
        }
    
    return pd.DataFrame(metrics).T


def build_models(choice: str, horizon: int, ma_window: int, n_lags: int, has_covariates: bool):
    """Build candidate models based on user selection
    
    Model covariate requirements:
    - No covariates: MovingAverage, VAR, ARIMA, ETS
    - Optional covariates: RandomForest, XGBoost, Prophet (use lags if no covariates)
    - Required covariates: Linear
    """
    if choice == "Auto (All Models)":
        # Base models that work without covariates
        models = [
            MovingAverageForecaster(horizon=horizon, window=ma_window),
            VARForecaster(horizon=horizon),
        ]
        
        # Add statistical models
        try:
            models.append(ARIMAForecaster(horizon=horizon))
            models.append(ETSForecaster(horizon=horizon))
        except Exception:
            pass
        
        # ML models work with or without covariates (use lags by default)
        try:
            models.extend([
                RandomForestForecaster(horizon=horizon, n_lags=n_lags),
                XGBoostForecaster(horizon=horizon, n_lags=n_lags),
            ])
        except Exception:
            pass
        
        # Linear and Prophet only if covariates available
        if has_covariates:
            try:
                models.append(LinearForecaster(horizon=horizon))
            except Exception:
                pass
            try:
                models.append(ProphetForecaster(horizon=horizon))
            except Exception:
                pass
        
        return models
    elif choice == "Auto (Base Models)":
        # Models that work without covariates
        models = [
            MovingAverageForecaster(horizon=horizon, window=ma_window),
            VARForecaster(horizon=horizon),
        ]
        # Add ML models (work with lags)
        try:
            models.append(RandomForestForecaster(horizon=horizon, n_lags=n_lags))
        except Exception:
            pass
        # Linear only if covariates
        if has_covariates:
            models.append(LinearForecaster(horizon=horizon))
        return models
    elif choice == "Moving Average":
        return [MovingAverageForecaster(horizon=horizon, window=ma_window)]
    elif choice == "Linear":
        if not has_covariates:
            st.error("❌ Linear model requires covariates. Please enable covariates or choose a different model (Moving Average, VAR, ARIMA, or ETS).")
            return [MovingAverageForecaster(horizon=horizon, window=ma_window)]
        return [LinearForecaster(horizon=horizon)]
    elif choice == "VAR":
        return [VARForecaster(horizon=horizon)]
    elif choice == "ARIMA":
        return [ARIMAForecaster(horizon=horizon)]
    elif choice == "ETS":
        return [ETSForecaster(horizon=horizon)]
    elif choice == "Prophet":
        if not has_covariates:
            st.info("ℹ️ Prophet works without covariates but may benefit from them. Proceeding without external regressors.")
        return [ProphetForecaster(horizon=horizon)]
    elif choice == "RandomForest":
        if not has_covariates:
            st.info("ℹ️ RandomForest will use lag features. Consider adding covariates for better accuracy.")
        return [RandomForestForecaster(horizon=horizon, n_lags=n_lags)]
    elif choice == "XGBoost":
        if not has_covariates:
            st.info("ℹ️ XGBoost will use lag features. Consider adding covariates for better accuracy.")
        return [XGBoostForecaster(horizon=horizon, n_lags=n_lags)]
    return [MovingAverageForecaster(horizon=horizon, window=ma_window)]


def forecast_series(ts: pd.DataFrame, horizon: int, model_choice: str, ma_window: int, n_lags: int,
                   X_train: Optional[pd.DataFrame] = None, X_test: Optional[pd.DataFrame] = None,
                   per_series_covariates: bool = False, covariate_mapping: Optional[Dict] = None,
                   backtest: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[Dict]]:
    """Run forecasting with optional backtesting and covariates"""
    if ts.empty:
        return None, None, None, None

    if backtest and len(ts) <= horizon:
        return None, None, None, None

    if backtest:
        y_train = ts.iloc[:-horizon]
        y_test = ts.iloc[-horizon:]
        if X_train is not None:
            if isinstance(X_train, dict):
                X_train_split = {k: v.iloc[:-horizon] for k, v in X_train.items()}
                X_test_split = {k: v.iloc[-horizon:] for k, v in X_train.items()}
            else:
                X_train_split = X_train.iloc[:-horizon]
                X_test_split = X_train.iloc[-horizon:]
        else:
            X_train_split = None
            X_test_split = None
    else:
        y_train = ts
        y_test = None
        X_train_split = X_train
        X_test_split = X_test

    has_covariates = X_train is not None
    
    # Validate covariate configuration
    if per_series_covariates and isinstance(X_train, dict):
        # Check if all target series have covariates
        missing_covs = [col for col in ts.columns if col not in X_train]
        if missing_covs:
            # This is OK - series without covariates will use models that don't need them
            pass
    
    candidates = build_models(model_choice, horizon, ma_window, n_lags, has_covariates)
    
    if not candidates:
        st.error("No valid models available for the current configuration.")
        return None, None, None, None

    auto = AutoForecaster(
        candidate_models=candidates,
        metric="rmse",
        n_splits=3,
        test_size=min(14, max(2, horizon // 2)),
        per_series_models=True,
        verbose=False,
    )

    auto.fit(y_train, X=X_train_split)
    forecast = auto.forecast(X=X_test_split)
    
    # Get selected models
    selected_models = {}
    if hasattr(auto, 'best_model_names_'):
        selected_models = auto.best_model_names_

    metrics = None
    if y_test is not None:
        common_cols = [c for c in y_test.columns if c in forecast.columns]
        y_test = y_test[common_cols]
        forecast_aligned = forecast[common_cols]
        metrics = calculate_metrics(y_test, forecast_aligned)

    return y_train, forecast, metrics, selected_models


def plot_forecast_comparison(history: pd.DataFrame, forecast: pd.DataFrame, 
                             title: str = "Forecast Results"):
    """Create enhanced forecast visualization"""
    n_series = len(history.columns)
    fig, axes = plt.subplots(n_series, 1, figsize=(12, 4 * n_series))
    
    if n_series == 1:
        axes = [axes]
    
    for idx, col in enumerate(history.columns):
        ax = axes[idx]
        
        # Plot historical data
        ax.plot(history.index, history[col], linewidth=2, 
               label=f"{col} (Historical)", color='#60A5FA', marker='o', 
               markersize=3, alpha=0.8)
        
        # Plot forecast
        if col in forecast.columns:
            ax.plot(forecast.index, forecast[col], linewidth=2.5, 
                   linestyle='--', label=f"{col} (Forecast)", 
                   color='#F87171', marker='s', markersize=4)
        
        ax.set_title(f"{col}", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    return fig


def plot_scenario_comparison(scenarios: Dict[str, pd.DataFrame]):
    """Plot multiple what-if scenarios"""
    if not scenarios:
        return None
    
    # Get first scenario to determine structure
    first_scenario = list(scenarios.values())[0]
    n_series = len(first_scenario.columns)
    
    fig, axes = plt.subplots(n_series, 1, figsize=(12, 4 * n_series))
    if n_series == 1:
        axes = [axes]
    
    colors = ['#60A5FA', '#F87171', '#34D399', '#FBBF24', '#A78BFA', '#F472B6']
    
    for idx, col in enumerate(first_scenario.columns):
        ax = axes[idx]
        
        for scenario_idx, (scenario_name, forecast) in enumerate(scenarios.items()):
            if col in forecast.columns:
                color = colors[scenario_idx % len(colors)]
                ax.plot(forecast.index, forecast[col], linewidth=2, 
                       label=scenario_name, color=color, marker='o', 
                       markersize=4, alpha=0.8)
        
        ax.set_title(f"{col} - Scenario Comparison", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    return fig


def show_data_format_guide():
    """Display expected data format documentation"""
    with st.expander("📋 Expected Input Data Format", expanded=False):
        st.markdown("""
        ### Required Format
        
        Your CSV file should have the following structure:
        
        | Column Type | Description | Example |
        |------------|-------------|---------|
        | **Date Column** | Datetime or string parseable as date | `2024-01-01`, `Jan 1, 2024` |
        | **Target Column(s)** | Numeric values to forecast | `123.45`, `1000` |
        
        ### Example Dataset Structure
        
        ```csv
        date,sales_product_a,sales_product_b,temperature,advertising_spend,competitor_price,promotion
        2024-01-01,1000,500,75.2,3200,18.5,0
        2024-01-02,1050,520,78.1,3500,19.2,1
        2024-01-03,980,510,72.3,2800,17.8,0
        ...
        ```
        
        ### With Covariates (Per-Series Example)
        - **Product A** could use: `temperature`, `advertising_spend`
        - **Product B** could use: `competitor_price`, `promotion`
        
        ### Tips
        - ✅ At least 30-50 data points recommended
        - ✅ Date column should be continuous (no large gaps)
        - ✅ Multiple series can be forecasted simultaneously
        - ✅ Missing values will be handled automatically
        - ❌ Avoid duplicate dates in your data
        """)
        
        # Sample data download
        sample_df = generate_sample_data()
        csv_buffer = BytesIO()
        sample_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Sample Dataset",
            data=csv_buffer,
            file_name="sample_timeseries.csv",
            mime="text/csv",
        )


def main():
    st.set_page_config(
        page_title="AutoTSForecast Studio", 
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "AutoTSForecast Studio - Advanced Time Series Forecasting Platform"
        }
    )
    
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #60A5FA; margin-bottom: 0;'>
            📈 AutoTSForecast Studio
        </h1>
        <p style='text-align: center; color: #9CA3AF; font-size: 18px;'>
            Advanced Time Series Forecasting with What-If Analysis
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/graph.png", width=80)
        st.markdown("### ⚙️ Configuration")
        
        # Data Source
        with st.expander("📂 Data Source", expanded=True):
            data_source = st.radio(
                "Choose data source",
                ["🎲 Sample Data", "📤 Upload CSV"],
                index=0,
                label_visibility="collapsed"
            )
            
            if data_source == "📤 Upload CSV":
                uploaded = st.file_uploader(
                    "Upload your CSV file",
                    type=["csv"],
                    help="Upload a CSV file with date and numeric columns"
                )
            else:
                uploaded = None
        
        # Model Configuration
        with st.expander("🤖 Model Settings", expanded=True):
            model_choice = st.selectbox(
                "Model Selection",
                ["Auto (All Models)", "Auto (Base Models)", "Moving Average", "Linear", 
                 "VAR", "ARIMA", "ETS", "Prophet", "RandomForest", "XGBoost"],
                help="Auto (All Models) includes ML models (requires covariates). Auto (Base Models) works without covariates."
            )
            
            horizon = st.slider(
                "Forecast Horizon (steps)",
                min_value=1, max_value=60, value=14,
                help="Number of future periods to forecast"
            )
            
            ma_window = st.slider(
                "Moving Average Window",
                min_value=2, max_value=30, value=7,
                help="Window size for moving average model"
            )
            
            n_lags = st.slider(
                "Number of Lags (ML models)",
                min_value=1, max_value=30, value=7,
                help="Number of lag features for ML models"
            )
            
            backtest = st.checkbox(
                "Enable Backtesting",
                value=True,
                help="Hold out last horizon periods for validation"
            )

    # Load Data
    if data_source == "🎲 Sample Data":
        df = generate_sample_data()
    else:
        if uploaded is None:
            show_data_format_guide()
            st.info("👆 Upload a CSV file to continue or use sample data")
            return
        df = pd.read_csv(uploaded)
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & Forecast", "🔮 What-If Analysis", "📈 Advanced Metrics", "💾 Export"])
    
    with tab1:
        # Data Preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(20), use_container_width=True, height=300)
        
        with col2:
            st.subheader("📊 Data Statistics")
            st.metric("Total Rows", len(df))
            st.metric("Total Columns", len(df.columns))
            
            date_col_guess = infer_date_column(df)
            if date_col_guess:
                st.metric("Date Column", date_col_guess)
        
        show_data_format_guide()
        
        # Column Selection
        st.subheader("🎯 Select Columns")
        col1, col2 = st.columns(2)
        
        with col1:
            date_col_guess = infer_date_column(df)
            date_col = st.selectbox(
                "📅 Date Column",
                options=list(df.columns),
                index=list(df.columns).index(date_col_guess) if date_col_guess in df.columns else 0,
                help="Select the column containing dates"
            )
        
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if date_col in numeric_cols:
                numeric_cols.remove(date_col)
            
            target_cols = st.multiselect(
                "🎯 Target Series",
                options=numeric_cols,
                default=numeric_cols[:2] if numeric_cols else [],
                help="Select one or more series to forecast"
            )
        
        # Covariate Configuration
        st.markdown("---")
        st.subheader("📊 Covariates Configuration (Optional)")
        
        with st.expander("ℹ️ What are Covariates?", expanded=False):
            st.markdown("""
            **Covariates** are additional features (external variables) that can help improve forecast accuracy.
            
            **Examples:**
            - 🌡️ Temperature data for ice cream sales
            - 📺 Ad spend for product demand
            - 💰 Competitor pricing for sales forecasting
            - 🎉 Promotion flags for retail
            
            **Per-Series Covariates** let you use different features for each target series:
            - Product A: temperature + advertising
            - Product B: competitor_price + promotions
            
            ---
            
            ### 📋 Model Covariate Support
            
            | Model | Covariates | Notes |
            |-------|-----------|-------|
            | **Moving Average** | ❌ Not used | Uses only historical values |
            | **VAR** | ❌ Not used | Multivariate autoregression |
            | **ARIMA** | ❌ Not used | Classical time series |
            | **ETS** | ❌ Not used | Exponential smoothing |
            | **Linear** | ✅ Required | Regression on external features |
            | **RandomForest** | ✅ Optional* | Works with lags + covariates |
            | **XGBoost** | ✅ Optional* | Works with lags + covariates |
            | **Prophet** | ✅ Optional | Can add external regressors |
            
            *ML models use lag features by default, but covariates significantly improve accuracy
            """)
        
        use_covariates = st.checkbox(
            "✅ Enable Covariates",
            value=False,
            help="Use additional features for forecasting (works best with ML models like RandomForest, XGBoost)"
        )
        
        covariate_cols = []
        per_series_covariates = False
        covariate_mapping = {}
        
        if use_covariates and target_cols:
            available_covariates = [col for col in numeric_cols if col not in target_cols]
            
            if not available_covariates:
                st.warning("⚠️ No additional numeric columns available for covariates. Add more columns to your dataset.")
            else:
                st.success(f"✅ {len(available_covariates)} covariate column(s) available: {', '.join(available_covariates)}")
                
                per_series_covariates = st.checkbox(
                    "🔀 Use Different Covariates per Series",
                    value=False,
                    help="Assign specific covariates to each target series independently"
                )
                
                if per_series_covariates:
                    st.markdown("### 🎯 Configure Covariates for Each Series")
                    st.markdown("*Select which features influence each target series:*")
                    
                    # Create columns for better layout
                    n_cols = min(len(target_cols), 3)
                    cols = st.columns(n_cols)
                    
                    for idx, target in enumerate(target_cols):
                        with cols[idx % n_cols]:
                            st.markdown(f"**{target}**")
                            selected = st.multiselect(
                                f"Features for {target}",
                                options=available_covariates,
                                key=f"cov_{target}",
                                help=f"Choose covariates that influence {target}",
                                label_visibility="collapsed"
                            )
                            if selected:
                                covariate_mapping[target] = selected
                                st.caption(f"✓ {len(selected)} covariate(s)")
                            else:
                                st.caption("⚠️ No covariates selected")
                    
                    # Summary table
                    if covariate_mapping:
                        st.markdown("#### 📋 Covariate Mapping Summary")
                        summary_data = []
                        for series, covs in covariate_mapping.items():
                            summary_data.append({
                                "Target Series": series,
                                "Covariates": ", ".join(covs),
                                "Count": len(covs)
                            })
                        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                    else:
                        st.info("👆 Select at least one covariate for one series to use per-series configuration")
                else:
                    st.markdown("### 📊 Shared Covariates")
                    st.markdown("*These features will be used for all target series:*")
                    covariate_cols = st.multiselect(
                        "Select Covariate Columns",
                        options=available_covariates,
                        help="Features that will be used for all target series"
                    )
                    
                    if covariate_cols:
                        st.info(f"✅ Using {len(covariate_cols)} shared covariate(s): {', '.join(covariate_cols)}")
                    else:
                        st.warning("⚠️ No covariates selected. Select at least one to improve forecasts.")
        
        if not target_cols:
            st.warning("⚠️ Please select at least one target column to forecast")
            return
        
        # Prepare time series
        ts = prepare_timeseries(df, date_col, target_cols)
        
        # Prepare covariates
        X_train = None
        X_test = None
        
        if use_covariates:
            if per_series_covariates and covariate_mapping:
                # Per-series covariates
                X_dict = {}
                for series, cov_cols in covariate_mapping.items():
                    if cov_cols:
                        X_series = prepare_timeseries(df, date_col, cov_cols)
                        X_dict[series] = X_series
                X_train = X_dict if X_dict else None
            elif covariate_cols:
                # Shared covariates
                X_train = prepare_timeseries(df, date_col, covariate_cols)
            
            if X_train is not None:
                st.info(f"✅ Covariates configured: {'Per-series' if per_series_covariates else 'Shared'}")
        
        # Run Forecast
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_forecast = st.button(
                "🚀 Run Forecast",
                type="primary",
                use_container_width=True
            )
        
        if run_forecast:
            with st.spinner("🔄 Training models and generating forecasts..."):
                history, forecast, metrics, selected_models = forecast_series(
                    ts, horizon, model_choice, ma_window, n_lags,
                    X_train=X_train, X_test=X_test,
                    per_series_covariates=per_series_covariates,
                    covariate_mapping=covariate_mapping,
                    backtest=backtest
                )
            
            if history is None or forecast is None:
                st.error("❌ Error generating forecast. Please check your data and parameters.")
                return
            
            # Store in session state
            st.session_state['history'] = history
            st.session_state['forecast'] = forecast
            st.session_state['metrics'] = metrics
            st.session_state['selected_models'] = selected_models
            st.session_state['ts'] = ts
            st.session_state['target_cols'] = target_cols
            st.session_state['horizon'] = horizon
            st.session_state['model_choice'] = model_choice
            st.session_state['ma_window'] = ma_window
            st.session_state['n_lags'] = n_lags
            st.session_state['X_train'] = X_train
            st.session_state['covariate_cols'] = covariate_cols
            st.session_state['per_series_covariates'] = per_series_covariates
            st.session_state['covariate_mapping'] = covariate_mapping
        
        # Display Results
        if 'forecast' in st.session_state:
            st.success("✅ Forecast completed successfully!")
            
            # Show covariate info
            if st.session_state.get('X_train') is not None:
                st.markdown("---")
                if st.session_state.get('per_series_covariates'):
                    st.info("🔀 **Per-Series Covariates Active**")
                    # Show detailed mapping
                    with st.expander("View Covariate Mapping Details"):
                        if st.session_state.get('covariate_mapping'):
                            for series, covs in st.session_state['covariate_mapping'].items():
                                st.markdown(f"**{series}:** {', '.join(covs)}")
                else:
                    cov_cols = st.session_state.get('covariate_cols', [])
                    if cov_cols:
                        st.info(f"📊 **Shared Covariates:** {', '.join(cov_cols)}")
                st.markdown("---")
            
            # Show selected models
            if st.session_state.get('selected_models'):
                st.subheader("🏆 Selected Models")
                model_df = pd.DataFrame([
                    {"Series": series, "Best Model": model}
                    for series, model in st.session_state['selected_models'].items()
                ])
                st.dataframe(model_df, use_container_width=True, hide_index=True)
            
            # Forecast output
            st.subheader("📊 Forecast Output")
            st.dataframe(st.session_state['forecast'], use_container_width=True)
            
            # Visualization
            st.subheader("📈 Forecast Visualization")
            fig = plot_forecast_comparison(
                st.session_state['history'],
                st.session_state['forecast']
            )
            st.pyplot(fig)
            
            # Metrics
            if st.session_state['metrics'] is not None:
                st.subheader("📉 Backtest Performance Metrics")
                st.dataframe(st.session_state['metrics'], use_container_width=True)
    
    with tab2:
        st.subheader("🔮 What-If Scenario Analysis")
        st.markdown("""
        Create and compare multiple forecast scenarios by adjusting parameters.
        
        **Note:** What-if analysis uses historical data only. Models that require future covariates (Linear, RandomForest with covariates, XGBoost with covariates) 
        will automatically switch to models that work without them (MA, VAR, ARIMA, ETS).
        """)
        
        if 'ts' not in st.session_state:
            st.info("👈 Please run a forecast first in the 'Data & Forecast' tab")
            return
        
        # Scenario builder
        st.markdown("### 🎛️ Build Scenarios")
        
        num_scenarios = st.number_input(
            "Number of scenarios to compare",
            min_value=2, max_value=5, value=3,
            help="Create multiple scenarios with different parameters"
        )
        
        scenarios = {}
        scenario_configs = []
        
        cols = st.columns(num_scenarios)
        
        for i in range(num_scenarios):
            with cols[i]:
                st.markdown(f"#### Scenario {i+1}")
                
                scenario_name = st.text_input(
                    "Name",
                    value=f"Scenario {i+1}",
                    key=f"scenario_name_{i}"
                )
                
                scenario_model = st.selectbox(
                    "Model",
                    ["Auto (Base Models)", "Moving Average", "VAR", "ARIMA", "ETS"],
                    key=f"scenario_model_{i}",
                    index=i % 5,
                    help="What-if analysis uses models that don't require future covariates"
                )
                
                scenario_horizon = st.slider(
                    "Horizon",
                    min_value=7, max_value=60, value=14 + (i * 7),
                    key=f"scenario_horizon_{i}"
                )
                
                scenario_ma = st.slider(
                    "MA Window",
                    min_value=2, max_value=30, value=7 + (i * 3),
                    key=f"scenario_ma_{i}"
                )
                
                scenario_configs.append({
                    'name': scenario_name,
                    'model': scenario_model,
                    'horizon': scenario_horizon,
                    'ma_window': scenario_ma
                })
        
        if st.button("🔄 Generate Scenario Forecasts", type="primary"):
            with st.spinner("Generating scenarios..."):
                n_lags_val = st.session_state.get('n_lags', 7)
                X_train_val = st.session_state.get('X_train')
                
                # For what-if analysis, we need to provide future covariates
                # Since we don't have true future values, we'll use the last available values
                # or skip models that require covariates
                has_covariates = X_train_val is not None
                
                if has_covariates:
                    st.warning("⚠️ What-if scenarios with covariates: Using last known covariate values for forecasting. For accurate results, provide future covariate values or use models that don't require covariates (MA, VAR, ARIMA, ETS).")
                
                for config in scenario_configs:
                    # For scenario analysis without future covariates, use models that don't need them
                    if has_covariates and config['model'] not in ["Moving Average", "VAR", "ARIMA", "ETS", "Auto (Base Models)"]:
                        # Skip or use only non-covariate models
                        st.info(f"ℹ️ Scenario '{config['name']}' with {config['model']}: Using models that don't require future covariates")
                    
                    _, forecast, _, _ = forecast_series(
                        st.session_state['ts'],
                        config['horizon'],
                        config['model'],
                        config['ma_window'],
                        n_lags_val,
                        X_train=None,  # Don't use covariates for what-if scenarios
                        X_test=None,
                        backtest=False
                    )
                    if forecast is not None:
                        scenarios[config['name']] = forecast
                
                st.session_state['scenarios'] = scenarios
        
        # Display scenario comparison
        if 'scenarios' in st.session_state and st.session_state['scenarios']:
            st.markdown("### 📊 Scenario Comparison")
            
            # Plot comparison
            fig = plot_scenario_comparison(st.session_state['scenarios'])
            if fig:
                st.pyplot(fig)
            
            # Scenario summary
            st.markdown("### 📋 Scenario Summary Statistics")
            
            summary_data = []
            for name, forecast in st.session_state['scenarios'].items():
                for col in forecast.columns:
                    summary_data.append({
                        'Scenario': name,
                        'Series': col,
                        'Mean': round(forecast[col].mean(), 2),
                        'Min': round(forecast[col].min(), 2),
                        'Max': round(forecast[col].max(), 2),
                        'Std Dev': round(forecast[col].std(), 2)
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("📈 Advanced Analytics")
        
        if 'history' not in st.session_state:
            st.info("👈 Please run a forecast first in the 'Data & Forecast' tab")
            return
        
        # Time series decomposition visualization
        st.markdown("### 📉 Time Series Statistics")
        
        history = st.session_state['history']
        
        stats_data = []
        for col in history.columns:
            stats_data.append({
                'Series': col,
                'Mean': round(history[col].mean(), 2),
                'Median': round(history[col].median(), 2),
                'Std Dev': round(history[col].std(), 2),
                'Min': round(history[col].min(), 2),
                'Max': round(history[col].max(), 2),
                'Trend': '📈' if history[col].iloc[-10:].mean() > history[col].iloc[:10].mean() else '📉'
            })
        
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        # Distribution plot
        st.markdown("### 📊 Value Distribution")
        fig, axes = plt.subplots(1, len(history.columns), figsize=(12, 4))
        if len(history.columns) == 1:
            axes = [axes]
        
        for idx, col in enumerate(history.columns):
            axes[idx].hist(history[col].dropna(), bins=30, alpha=0.7, color='#60A5FA', edgecolor='white')
            axes[idx].set_title(f"{col} Distribution", fontsize=11, fontweight='bold')
            axes[idx].set_xlabel("Value", fontsize=9)
            axes[idx].set_ylabel("Frequency", fontsize=9)
            axes[idx].grid(True, alpha=0.2)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab4:
        st.subheader("💾 Export Results")
        
        if 'forecast' not in st.session_state:
            st.info("👈 Please run a forecast first in the 'Data & Forecast' tab")
            return
        
        st.markdown("### 📥 Download Forecast Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download forecast
            forecast_csv = st.session_state['forecast'].to_csv()
            st.download_button(
                label="📊 Download Forecast CSV",
                data=forecast_csv,
                file_name="forecast_results.csv",
                mime="text/csv",
            )
        
        with col2:
            # Download metrics
            if st.session_state.get('metrics') is not None:
                metrics_csv = st.session_state['metrics'].to_csv()
                st.download_button(
                    label="📉 Download Metrics CSV",
                    data=metrics_csv,
                    file_name="forecast_metrics.csv",
                    mime="text/csv",
                )
        
        # Download scenarios
        if 'scenarios' in st.session_state and st.session_state['scenarios']:
            st.markdown("### 🔮 Download Scenario Results")
            
            for name, forecast in st.session_state['scenarios'].items():
                scenario_csv = forecast.to_csv()
                st.download_button(
                    label=f"📊 Download {name}",
                    data=scenario_csv,
                    file_name=f"scenario_{name.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    key=f"download_{name}"
                )


if __name__ == "__main__":
    main()
