"""
Quickstart Example for TS-Forecast Package

This example demonstrates the complete workflow:
1. Model Selection
2. Backtesting
3. Hierarchical Forecasting
4. Driver Analysis (Interpretability)
"""

import numpy as np
import pandas as pd
from ts_forecast import (
    ModelSelector,
    BacktestValidator,
    HierarchicalReconciler,
    DriverAnalyzer,
    VARForecaster,
    LinearForecaster
)
from ts_forecast.utils import preprocess_data, split_data, create_time_series_features


def generate_sample_data(n_periods=200):
    """Generate sample multivariate time series data"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    # Generate correlated time series
    trend = np.linspace(0, 20, n_periods)
    seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, n_periods))
    
    data = pd.DataFrame({
        'sales_A': trend + seasonality + np.random.randn(n_periods) * 2 + 100,
        'sales_B': trend * 0.6 + seasonality * 0.8 + np.random.randn(n_periods) * 1.5 + 80,
        'sales_C': trend * 0.4 + np.random.randn(n_periods) * 1.2 + 60
    }, index=dates)
    
    # Add some covariates
    covariates = pd.DataFrame({
        'price': 20 + np.sin(np.linspace(0, 2*np.pi, n_periods)) * 5 + np.random.randn(n_periods),
        'promotion': np.random.choice([0, 1], n_periods, p=[0.7, 0.3]),
        'temperature': 20 + 10 * np.sin(np.linspace(0, 4*np.pi, n_periods)) + np.random.randn(n_periods) * 3
    }, index=dates)
    
    return data, covariates


def example_model_selection():
    """Example 1: Model Selection"""
    print("\n" + "="*60)
    print("EXAMPLE 1: MODEL SELECTION")
    print("="*60)
    
    # Generate data
    data, _ = generate_sample_data(n_periods=150)
    
    # Preprocess
    data = preprocess_data(data, handle_missing='forward_fill')
    
    # Model selection
    print("\nPerforming model selection...")
    selector = ModelSelector(metric='rmse')
    selector.fit(data, validation_split=0.2, cv_folds=3)
    
    # Results
    print("\nModel Performance:")
    results = selector.get_results()
    for model_name, result in results.items():
        if isinstance(result, dict):
            print(f"  {model_name}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
        else:
            print(f"  {model_name}: {result:.4f}")
    
    best_name, best_model = selector.get_best_model()
    print(f"\nBest Model: {best_name}")
    
    # Generate forecast
    forecasts = selector.predict()
    print(f"\nForecast shape: {forecasts.shape}")
    print("First forecast:")
    print(forecasts.head())
    
    return selector


def example_backtesting():
    """Example 2: Backtesting"""
    print("\n" + "="*60)
    print("EXAMPLE 2: BACKTESTING")
    print("="*60)
    
    # Generate data
    data, _ = generate_sample_data(n_periods=150)
    data = preprocess_data(data)
    
    # Create model
    model = VARForecaster(horizon=1, lags=3)
    
    # Run backtesting
    print("\nRunning backtesting with 5 folds...")
    validator = BacktestValidator(
        model=model,
        n_splits=5,
        test_size=10,
        window_type='expanding'
    )
    
    overall_metrics = validator.run(data)
    
    # Display results
    print("\nOverall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nFold-by-Fold Results:")
    fold_results = validator.get_fold_results()
    print(fold_results[['fold', 'train_size', 'test_size', 'rmse', 'mape', 'r2']])
    
    print("\nSummary Statistics:")
    print(validator.get_summary())
    
    return validator


def example_hierarchical_forecasting():
    """Example 3: Hierarchical Forecasting"""
    print("\n" + "="*60)
    print("EXAMPLE 3: HIERARCHICAL FORECASTING")
    print("="*60)
    
    # Define hierarchy
    hierarchy = {
        'Total': ['Region_A', 'Region_B'],
        'Region_A': ['Store_A1', 'Store_A2'],
        'Region_B': ['Store_B1', 'Store_B2', 'Store_B3']
    }
    
    # Generate base forecasts (normally from your models)
    np.random.seed(42)
    n_periods = 10
    forecasts = pd.DataFrame({
        'Total': 1000 + np.cumsum(np.random.randn(n_periods) * 10),
        'Region_A': 600 + np.cumsum(np.random.randn(n_periods) * 6),
        'Region_B': 400 + np.cumsum(np.random.randn(n_periods) * 5),
        'Store_A1': 350 + np.cumsum(np.random.randn(n_periods) * 3),
        'Store_A2': 250 + np.cumsum(np.random.randn(n_periods) * 3),
        'Store_B1': 150 + np.cumsum(np.random.randn(n_periods) * 2),
        'Store_B2': 130 + np.cumsum(np.random.randn(n_periods) * 2),
        'Store_B3': 120 + np.cumsum(np.random.randn(n_periods) * 2),
    })
    
    print("\nBase Forecasts (first 5 periods):")
    print(forecasts.head())
    
    # Reconcile using different methods
    methods = ['bottom_up', 'top_down', 'ols']
    
    for method in methods:
        print(f"\n--- Reconciliation Method: {method.upper()} ---")
        reconciler = HierarchicalReconciler(forecasts, hierarchy)
        reconciler.reconcile(method=method)
        
        reconciled = reconciler.get_reconciled_forecasts()
        is_coherent = reconciler.validate_coherency()
        
        print(f"Coherency Check: {'✓ PASS' if is_coherent else '✗ FAIL'}")
        print(f"Reconciled Forecasts (first 3 periods):")
        print(reconciled.head(3))
        
        # Show reconciliation info
        info = reconciler.get_reconciliation_info()
        print(f"Bottom-level series: {info['n_bottom_series']}")
        print(f"Aggregated series: {info['n_aggregated_series']}")


def example_driver_analysis():
    """Example 4: Driver Analysis (Interpretability)"""
    print("\n" + "="*60)
    print("EXAMPLE 4: DRIVER ANALYSIS")
    print("="*60)
    
    # Generate data with covariates
    data, covariates = generate_sample_data(n_periods=150)
    
    # Split data
    train_size = 120
    y_train = data.iloc[:train_size]
    y_test = data.iloc[train_size:]
    X_train = covariates.iloc[:train_size]
    X_test = covariates.iloc[train_size:]
    
    # Fit model with covariates
    print("\nFitting LinearForecaster with covariates...")
    model = LinearForecaster(horizon=1)
    model.fit(y_train, X_train)
    
    # Analyze drivers
    print("\nAnalyzing feature importance...")
    analyzer = DriverAnalyzer(model, feature_names=X_train.columns.tolist())
    
    # Coefficient importance
    coef_importance = analyzer.calculate_feature_importance(X_train, y_train, method='coefficients')
    print("\nCoefficient Importance:")
    print(coef_importance)
    
    # Sensitivity analysis
    sensitivity = analyzer.calculate_feature_importance(X_train, y_train, method='sensitivity')
    print("\nSensitivity Analysis:")
    print(sensitivity)
    
    # Comprehensive analysis
    print("\nComprehensive Driver Analysis:")
    analysis = analyzer.analyze_drivers(
        X_train, y_train,
        numerical_features=['price', 'temperature'],
        categorical_features=['promotion']
    )
    
    print(f"  Numerical features: {analysis['numerical_features']}")
    print(f"  Categorical features: {analysis['categorical_features']}")
    
    return analyzer


def main():
    """Run all examples"""
    print("\n" + "#"*60)
    print("# TS-FORECAST MVP - QUICKSTART EXAMPLES")
    print("#"*60)
    
    # Run all examples
    selector = example_model_selection()
    validator = example_backtesting()
    example_hierarchical_forecasting()
    analyzer = example_driver_analysis()
    
    print("\n" + "#"*60)
    print("# ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("#"*60)
    print("\nNext steps:")
    print("  1. Try with your own data")
    print("  2. Experiment with different models and parameters")
    print("  3. Explore visualization options with .plot_results()")
    print("  4. Check out the tests/ directory for more examples")


if __name__ == "__main__":
    main()