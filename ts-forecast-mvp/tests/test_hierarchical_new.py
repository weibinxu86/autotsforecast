"""Tests for hierarchical reconciliation"""

import numpy as np
import pandas as pd
import pytest
from ts_forecast.hierarchical.reconciliation import HierarchicalReconciler


@pytest.fixture
def sample_hierarchy():
    """Sample hierarchy structure"""
    return {
        'Total': ['A', 'B'],
        'A': ['A1', 'A2'],
        'B': ['B1', 'B2']
    }


@pytest.fixture
def sample_forecasts():
    """Sample forecasts for all levels"""
    np.random.seed(42)
    return pd.DataFrame({
        'Total': [100, 110, 120],
        'A': [60, 65, 70],
        'B': [40, 45, 50],
        'A1': [35, 38, 40],
        'A2': [25, 27, 30],
        'B1': [22, 25, 28],
        'B2': [18, 20, 22]
    })


class TestHierarchicalReconciler:
    """Tests for Hierarchical Reconciler"""
    
    def test_initialization(self, sample_forecasts, sample_hierarchy):
        """Test reconciler initialization"""
        reconciler = HierarchicalReconciler(sample_forecasts, sample_hierarchy)
        
        assert reconciler.forecasts.shape == sample_forecasts.shape
        assert reconciler.summing_matrix is not None
        assert reconciler.bottom_series is not None
        assert reconciler.aggregated_series is not None
    
    def test_bottom_level_identification(self, sample_forecasts, sample_hierarchy):
        """Test bottom level series identification"""
        reconciler = HierarchicalReconciler(sample_forecasts, sample_hierarchy)
        
        assert set(reconciler.bottom_series) == {'A1', 'A2', 'B1', 'B2'}
        assert 'Total' in reconciler.aggregated_series
        assert 'A' in reconciler.aggregated_series
        assert 'B' in reconciler.aggregated_series
    
    def test_bottom_up_reconciliation(self, sample_forecasts, sample_hierarchy):
        """Test bottom-up reconciliation"""
        reconciler = HierarchicalReconciler(sample_forecasts, sample_hierarchy)
        reconciler.reconcile(method='bottom_up')
        
        reconciled = reconciler.get_reconciled_forecasts()
        assert reconciled.shape[0] == sample_forecasts.shape[0]
        assert reconciler.validate_coherency()
    
    def test_top_down_reconciliation(self, sample_forecasts, sample_hierarchy):
        """Test top-down reconciliation"""
        reconciler = HierarchicalReconciler(sample_forecasts, sample_hierarchy)
        reconciler.reconcile(method='top_down')
        
        reconciled = reconciler.get_reconciled_forecasts()
        assert reconciled.shape[0] == sample_forecasts.shape[0]
        assert reconciler.validate_coherency()
    
    def test_ols_reconciliation(self, sample_forecasts, sample_hierarchy):
        """Test OLS (MinT) reconciliation"""
        reconciler = HierarchicalReconciler(sample_forecasts, sample_hierarchy)
        reconciler.reconcile(method='ols')
        
        reconciled = reconciler.get_reconciled_forecasts()
        assert reconciled.shape[0] == sample_forecasts.shape[0]
        # OLS should produce coherent forecasts
        assert reconciler.validate_coherency()
    
    def test_coherency_validation(self, sample_forecasts, sample_hierarchy):
        """Test coherency validation"""
        reconciler = HierarchicalReconciler(sample_forecasts, sample_hierarchy)
        reconciler.reconcile(method='bottom_up')
        
        # Check that sum of children equals parent
        reconciled = reconciler.get_reconciled_forecasts()
        
        # Check Total = A + B
        assert np.allclose(
            reconciled['Total'].values,
            reconciled['A'].values + reconciled['B'].values,
            rtol=1e-5
        )
        
        # Check A = A1 + A2
        assert np.allclose(
            reconciled['A'].values,
            reconciled['A1'].values + reconciled['A2'].values,
            rtol=1e-5
        )
    
    def test_reconciliation_info(self, sample_forecasts, sample_hierarchy):
        """Test getting reconciliation info"""
        reconciler = HierarchicalReconciler(sample_forecasts, sample_hierarchy)
        reconciler.reconcile(method='bottom_up')
        
        info = reconciler.get_reconciliation_info()
        
        assert info['n_bottom_series'] == 4
        assert info['n_aggregated_series'] == 3
        assert info['n_total_series'] == 7
        assert info['is_coherent'] is True
    
    def test_invalid_hierarchy(self, sample_forecasts):
        """Test with invalid hierarchy (missing series)"""
        invalid_hierarchy = {
            'Total': ['A', 'B'],
            'A': ['A1', 'A2'],
            'B': ['B1', 'MISSING']  # This series doesn't exist
        }
        
        with pytest.raises(ValueError):
            HierarchicalReconciler(sample_forecasts, invalid_hierarchy)
    
    def test_invalid_method(self, sample_forecasts, sample_hierarchy):
        """Test with invalid reconciliation method"""
        reconciler = HierarchicalReconciler(sample_forecasts, sample_hierarchy)
        
        with pytest.raises(ValueError):
            reconciler.reconcile(method='invalid_method')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
