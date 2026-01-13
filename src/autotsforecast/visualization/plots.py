"""
Visualization tools for time series forecasting.

Provides both static (matplotlib) and interactive (plotly) visualizations
for forecasts, model comparison, and diagnostic plots.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Any


# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def plot_forecast(
    y_train: pd.DataFrame,
    forecasts: pd.DataFrame,
    y_test: Optional[pd.DataFrame] = None,
    intervals: Optional[Dict[str, pd.DataFrame]] = None,
    series: Optional[Union[str, List[str]]] = None,
    title: str = "Forecast",
    figsize: tuple = (14, 6),
    show_legend: bool = True,
    ax: Optional[Any] = None
) -> Any:
    """
    Plot forecasts with optional prediction intervals.
    
    Parameters
    ----------
    y_train : pd.DataFrame
        Historical training data
    forecasts : pd.DataFrame  
        Forecasted values
    y_test : pd.DataFrame, optional
        Actual test values for comparison
    intervals : dict, optional
        Prediction intervals from PredictionIntervals.predict()
        Should contain keys like 'lower_95', 'upper_95'
    series : str or list of str, optional
        Specific series to plot. If None, plots all.
    title : str
        Plot title
    figsize : tuple
        Figure size
    show_legend : bool
        Whether to show legend
    ax : matplotlib axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib figure or axes
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Determine series to plot
    if series is None:
        series_list = y_train.columns.tolist()
    elif isinstance(series, str):
        series_list = [series]
    else:
        series_list = series
    
    # Create figure
    n_series = len(series_list)
    if n_series > 1:
        n_cols = min(2, n_series)
        n_rows = (n_series + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows // 2))
        if n_series == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        fig = ax.figure
    
    colors = plt.cm.tab10.colors
    
    for idx, col in enumerate(series_list):
        ax = axes[idx]
        color = colors[idx % len(colors)]
        
        # Plot training data
        if col in y_train.columns:
            ax.plot(y_train.index, y_train[col], 
                   color=color, alpha=0.7, label='Historical')
        
        # Plot forecasts
        if col in forecasts.columns:
            ax.plot(forecasts.index, forecasts[col],
                   color=color, linestyle='--', linewidth=2, label='Forecast')
        
        # Plot prediction intervals
        if intervals is not None:
            # Find all coverage levels
            coverages = set()
            for key in intervals.keys():
                if key.startswith('lower_'):
                    cov = key.split('_')[1]
                    coverages.add(cov)
            
            coverages = sorted(coverages, reverse=True)
            alphas = np.linspace(0.1, 0.3, len(coverages))
            
            for i, cov in enumerate(coverages):
                lower_key = f'lower_{cov}'
                upper_key = f'upper_{cov}'
                
                if lower_key in intervals and upper_key in intervals:
                    lower = intervals[lower_key]
                    upper = intervals[upper_key]
                    
                    if col in lower.columns and col in upper.columns:
                        ax.fill_between(
                            forecasts.index,
                            lower[col],
                            upper[col],
                            alpha=alphas[i],
                            color=color,
                            label=f'{cov}% CI'
                        )
        
        # Plot test data
        if y_test is not None and col in y_test.columns:
            ax.plot(y_test.index, y_test[col],
                   color='black', alpha=0.5, linewidth=1, label='Actual')
        
        ax.set_title(f'{col}' if n_series > 1 else title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        
        if show_legend:
            ax.legend(loc='best')
        
        # Format x-axis for dates
        if isinstance(y_train.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Hide empty subplots
    for idx in range(len(series_list), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title if n_series > 1 else '', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_forecast_interactive(
    y_train: pd.DataFrame,
    forecasts: pd.DataFrame,
    y_test: Optional[pd.DataFrame] = None,
    intervals: Optional[Dict[str, pd.DataFrame]] = None,
    series: Optional[Union[str, List[str]]] = None,
    title: str = "Interactive Forecast"
) -> Any:
    """
    Create interactive Plotly forecast visualization.
    
    Parameters
    ----------
    y_train : pd.DataFrame
        Historical training data
    forecasts : pd.DataFrame
        Forecasted values
    y_test : pd.DataFrame, optional
        Actual test values
    intervals : dict, optional
        Prediction intervals
    series : str or list of str, optional
        Series to plot
    title : str
        Plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for interactive plots. Install with: pip install plotly")
    
    # Determine series
    if series is None:
        series_list = y_train.columns.tolist()
    elif isinstance(series, str):
        series_list = [series]
    else:
        series_list = series
    
    n_series = len(series_list)
    
    # Create subplots
    fig = make_subplots(
        rows=n_series, 
        cols=1,
        subplot_titles=series_list,
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.T10
    
    for idx, col in enumerate(series_list):
        row = idx + 1
        color = colors[idx % len(colors)]
        
        # Historical data
        if col in y_train.columns:
            fig.add_trace(
                go.Scatter(
                    x=y_train.index,
                    y=y_train[col],
                    name=f'{col} - Historical',
                    line=dict(color=color),
                    legendgroup=col,
                    showlegend=True
                ),
                row=row, col=1
            )
        
        # Prediction intervals (add before forecast line)
        if intervals is not None:
            coverages = set()
            for key in intervals.keys():
                if key.startswith('lower_'):
                    cov = key.split('_')[1]
                    coverages.add(cov)
            
            coverages = sorted(coverages, reverse=True)
            
            for cov in coverages:
                lower_key = f'lower_{cov}'
                upper_key = f'upper_{cov}'
                
                if lower_key in intervals and upper_key in intervals:
                    lower = intervals[lower_key]
                    upper = intervals[upper_key]
                    
                    if col in lower.columns and col in upper.columns:
                        # Add filled area
                        fig.add_trace(
                            go.Scatter(
                                x=list(forecasts.index) + list(forecasts.index)[::-1],
                                y=list(upper[col]) + list(lower[col])[::-1],
                                fill='toself',
                                fillcolor=f'rgba({",".join(str(int(c*255)) for c in px.colors.hex_to_rgb(color))},0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name=f'{col} - {cov}% CI',
                                legendgroup=col,
                                showlegend=True,
                                hoverinfo='skip'
                            ),
                            row=row, col=1
                        )
        
        # Forecast
        if col in forecasts.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecasts.index,
                    y=forecasts[col],
                    name=f'{col} - Forecast',
                    line=dict(color=color, dash='dash', width=2),
                    legendgroup=col,
                    showlegend=True
                ),
                row=row, col=1
            )
        
        # Actual test data
        if y_test is not None and col in y_test.columns:
            fig.add_trace(
                go.Scatter(
                    x=y_test.index,
                    y=y_test[col],
                    name=f'{col} - Actual',
                    line=dict(color='black', width=1),
                    legendgroup=col,
                    showlegend=True
                ),
                row=row, col=1
            )
    
    fig.update_layout(
        title=title,
        height=300 * n_series,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig


def plot_model_comparison(
    cv_results: Dict[str, Dict[str, float]],
    metric: str = 'rmse',
    title: str = "Model Comparison",
    figsize: tuple = (12, 6),
    interactive: bool = False
) -> Any:
    """
    Compare model performance from cross-validation results.
    
    Parameters
    ----------
    cv_results : dict
        Cross-validation results from AutoForecaster.cv_results_
    metric : str
        Metric to plot ('rmse', 'mae', 'mape', 'r2')
    title : str
        Plot title
    figsize : tuple
        Figure size for static plot
    interactive : bool
        Whether to use Plotly for interactive plot
        
    Returns
    -------
    figure
    """
    # Extract metric values
    model_names = list(cv_results.keys())
    metric_values = [cv_results[m].get(metric, 0) for m in model_names]
    
    # Sort by performance
    if metric == 'r2':
        sorted_idx = np.argsort(metric_values)[::-1]  # Higher is better
    else:
        sorted_idx = np.argsort(metric_values)  # Lower is better
    
    model_names = [model_names[i] for i in sorted_idx]
    metric_values = [metric_values[i] for i in sorted_idx]
    
    if interactive and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=metric_values,
                marker_color=['green' if i == 0 else 'steelblue' for i in range(len(model_names))]
            )
        ])
        fig.update_layout(
            title=title,
            xaxis_title='Model',
            yaxis_title=metric.upper(),
            xaxis_tickangle=-45
        )
        return fig
    
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if i == 0 else 'steelblue' for i in range(len(model_names))]
    bars = ax.bar(range(len(model_names)), metric_values, color=colors)
    
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Highlight best model
    ax.bar(0, metric_values[0], color='green', label='Best Model')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance: Dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: tuple = (10, 8),
    interactive: bool = False
) -> Any:
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters
    ----------
    importance : dict
        Feature importance scores
    title : str
        Plot title
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    interactive : bool
        Use Plotly
        
    Returns
    -------
    figure
    """
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    if interactive and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Bar(
                y=features[::-1],
                x=values[::-1],
                orientation='h',
                marker_color='steelblue'
            )
        ])
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=max(400, 25 * len(features))
        )
        return fig
    
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = range(len(features))
    colors = ['green' if v >= 0 else 'red' for v in values]
    
    ax.barh(y_pos, values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def plot_residuals(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    series: Optional[str] = None,
    figsize: tuple = (14, 10)
) -> Any:
    """
    Diagnostic residual plots.
    
    Creates a 2x2 grid with:
    - Time series of residuals
    - Histogram of residuals
    - Q-Q plot
    - Residuals vs fitted values
    
    Parameters
    ----------
    y_true : pd.DataFrame
        Actual values
    y_pred : pd.DataFrame
        Predicted values
    series : str, optional
        Specific series to analyze. If None, uses first column.
    figsize : tuple
        Figure size
        
    Returns
    -------
    figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    from scipy import stats as scipy_stats
    
    if series is None:
        series = y_true.columns[0]
    
    residuals = y_true[series].values - y_pred[series].values
    fitted = y_pred[series].values
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Time series of residuals
    ax1 = axes[0, 0]
    ax1.plot(y_true.index, residuals, 'b-', alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('Residuals Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residual')
    
    # 2. Histogram
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue')
    
    # Fit normal distribution
    mu, std = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    ax2.plot(x, scipy_stats.norm.pdf(x, mu, std), 'r-', linewidth=2)
    ax2.set_title(f'Residual Distribution (μ={mu:.2f}, σ={std:.2f})')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Density')
    
    # 3. Q-Q plot
    ax3 = axes[1, 0]
    scipy_stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal)')
    
    # 4. Residuals vs Fitted
    ax4 = axes[1, 1]
    ax4.scatter(fitted, residuals, alpha=0.5, color='steelblue')
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_title('Residuals vs Fitted')
    ax4.set_xlabel('Fitted Values')
    ax4.set_ylabel('Residual')
    
    # Add lowess smoothing if seaborn available
    if SEABORN_AVAILABLE:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, fitted, frac=0.3)
            ax4.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2)
        except:
            pass
    
    plt.suptitle(f'Residual Diagnostics: {series}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_components(
    y: pd.DataFrame,
    series: Optional[str] = None,
    period: Optional[int] = None,
    figsize: tuple = (14, 10)
) -> Any:
    """
    Decompose and plot time series components.
    
    Parameters
    ----------
    y : pd.DataFrame
        Time series data
    series : str, optional
        Series to decompose
    period : int, optional
        Seasonal period. Auto-detected if None.
    figsize : tuple
        Figure size
        
    Returns
    -------
    figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required")
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    if series is None:
        series = y.columns[0]
    
    ts = y[series].dropna()
    
    # Auto-detect period
    if period is None:
        if isinstance(ts.index, pd.DatetimeIndex):
            freq = pd.infer_freq(ts.index)
            if freq in ['D', 'B']:
                period = 7  # Weekly seasonality
            elif freq in ['W', 'W-SUN', 'W-MON']:
                period = 52  # Yearly seasonality
            elif freq in ['M', 'MS']:
                period = 12  # Yearly seasonality
            elif freq in ['H']:
                period = 24  # Daily seasonality
            else:
                period = 7
        else:
            period = 7
    
    # Decompose
    decomposition = seasonal_decompose(ts, model='additive', period=period)
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    axes[0].plot(ts.index, decomposition.observed, 'b-')
    axes[0].set_title('Original')
    axes[0].set_ylabel('Value')
    
    axes[1].plot(ts.index, decomposition.trend, 'g-')
    axes[1].set_title('Trend')
    axes[1].set_ylabel('Value')
    
    axes[2].plot(ts.index, decomposition.seasonal, 'r-')
    axes[2].set_title(f'Seasonal (period={period})')
    axes[2].set_ylabel('Value')
    
    axes[3].plot(ts.index, decomposition.resid, 'purple', alpha=0.7)
    axes[3].set_title('Residual')
    axes[3].set_ylabel('Value')
    axes[3].set_xlabel('Date')
    
    plt.suptitle(f'Time Series Decomposition: {series}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


class ForecastPlotter:
    """
    Unified interface for forecast visualization.
    
    Provides convenient methods for common visualization tasks.
    
    Parameters
    ----------
    interactive : bool, default=False
        Use Plotly for interactive plots
    style : str, default='default'
        Matplotlib style (e.g., 'seaborn', 'ggplot', 'dark_background')
        
    Examples
    --------
    >>> plotter = ForecastPlotter(interactive=True)
    >>> plotter.plot_forecast(y_train, forecasts, intervals=intervals)
    >>> plotter.plot_comparison(cv_results)
    """
    
    def __init__(self, interactive: bool = False, style: str = 'default'):
        self.interactive = interactive
        self.style = style
        
        if MATPLOTLIB_AVAILABLE and style != 'default':
            try:
                plt.style.use(style)
            except:
                pass
    
    def plot_forecast(self, *args, **kwargs):
        """Plot forecasts."""
        if self.interactive:
            return plot_forecast_interactive(*args, **kwargs)
        return plot_forecast(*args, **kwargs)
    
    def plot_comparison(self, *args, **kwargs):
        """Plot model comparison."""
        return plot_model_comparison(*args, interactive=self.interactive, **kwargs)
    
    def plot_importance(self, *args, **kwargs):
        """Plot feature importance."""
        return plot_feature_importance(*args, interactive=self.interactive, **kwargs)
    
    def plot_residuals(self, *args, **kwargs):
        """Plot residual diagnostics."""
        return plot_residuals(*args, **kwargs)
    
    def plot_components(self, *args, **kwargs):
        """Plot time series components."""
        return plot_components(*args, **kwargs)
    
    def show(self):
        """Display the current plot."""
        if MATPLOTLIB_AVAILABLE:
            plt.show()
