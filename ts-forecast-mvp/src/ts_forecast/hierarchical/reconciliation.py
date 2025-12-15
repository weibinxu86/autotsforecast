import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.linalg import lstsq


class HierarchicalReconciler:
    """Hierarchical forecast reconciliation using multiple methods"""
    
    def __init__(self, forecasts: pd.DataFrame, hierarchy: Dict[str, List[str]]):
        """
        Args:
            forecasts: DataFrame with base forecasts for all series (rows=time, cols=series)
            hierarchy: Dict mapping parent nodes to their children
                      Example: {'Total': ['A', 'B'], 'A': ['A1', 'A2'], 'B': ['B1', 'B2']}
        """
        self.forecasts = forecasts
        self.hierarchy = hierarchy
        self.summing_matrix = None
        self.reconciled_forecasts = None
        self.bottom_series = None
        self.aggregated_series = None
        self._validate_hierarchy()
        self._build_summing_matrix()
        
    def _validate_hierarchy(self):
        """Validate the hierarchy structure"""
        all_series = set(self.forecasts.columns)
        
        # Check all hierarchy nodes exist in forecasts
        hierarchy_nodes = set(self.hierarchy.keys())
        for children in self.hierarchy.values():
            hierarchy_nodes.update(children)
        
        missing = hierarchy_nodes - all_series
        if missing:
            raise ValueError(f"Hierarchy nodes not in forecasts: {missing}")
        
        # Check for cycles
        visited = set()
        
        def has_cycle(node, path):
            if node in path:
                return True
            if node not in self.hierarchy:
                return False
            path.add(node)
            for child in self.hierarchy[node]:
                if has_cycle(child, path.copy()):
                    return True
            return False
        
        for node in self.hierarchy.keys():
            if has_cycle(node, set()):
                raise ValueError(f"Cycle detected in hierarchy starting at {node}")
    
    def _build_summing_matrix(self):
        """Build the summing matrix S for the hierarchy"""
        # Get all bottom-level series (leaf nodes)
        all_series = set(self.forecasts.columns)
        self.bottom_series = sorted(self._get_bottom_level_series())
        self.aggregated_series = sorted(all_series - set(self.bottom_series))
        
        n_agg = len(self.aggregated_series)
        n_bottom = len(self.bottom_series)
        
        if n_bottom == 0:
            raise ValueError("No bottom-level series found in hierarchy")
        
        # S = [S_agg; I] where S_agg defines aggregation relationships
        S_agg = np.zeros((n_agg, n_bottom))
        
        for i, agg_name in enumerate(self.aggregated_series):
            descendants = self._get_all_descendants(agg_name)
            for desc in descendants:
                if desc in self.bottom_series:
                    j = self.bottom_series.index(desc)
                    S_agg[i, j] = 1
        
        # Full summing matrix: [aggregated series; bottom series]
        self.summing_matrix = np.vstack([S_agg, np.eye(n_bottom)])
        
    def _get_bottom_level_series(self) -> List[str]:
        """Identify bottom-level series (leaf nodes with no children)"""
        all_children = set()
        for children in self.hierarchy.values():
            all_children.update(children)
        
        parents = set(self.hierarchy.keys())
        bottom = all_children - parents
        
        # Also include any series not in hierarchy
        all_series = set(self.forecasts.columns)
        bottom.update(all_series - parents - all_children)
        
        return list(bottom)
    
    def _get_all_descendants(self, node: str) -> List[str]:
        """Get all descendant nodes recursively"""
        if node not in self.hierarchy:
            return [node]
        
        descendants = []
        for child in self.hierarchy[node]:
            descendants.extend(self._get_all_descendants(child))
        
        return descendants
    
    def reconcile(self, method: str = 'bottom_up') -> 'HierarchicalReconciler':
        """Reconcile forecasts to satisfy hierarchical constraints
        
        Args:
            method: Reconciliation method
                - 'bottom_up': Sum from bottom level
                - 'top_down': Distribute from top using proportions
                - 'middle_out': Combine top-down and bottom-up
                - 'mint_sample': MinT with sample covariance (OLS)
                - 'ols': Ordinary Least Squares reconciliation
        
        Returns:
            self
        """
        if method == 'bottom_up':
            self._reconcile_bottom_up()
        elif method == 'top_down':
            self._reconcile_top_down()
        elif method == 'middle_out':
            self._reconcile_middle_out()
        elif method in ['mint_sample', 'ols']:
            self._reconcile_mint_ols()
        else:
            raise ValueError(f"Unknown reconciliation method: {method}")
        
        return self
    
    def _reconcile_bottom_up(self):
        """Bottom-up reconciliation: aggregate from bottom level"""
        bottom_forecasts = self.forecasts[self.bottom_series].values
        
        # Reconciled = S @ bottom_forecasts^T
        reconciled = self.summing_matrix @ bottom_forecasts.T
        
        all_series = self.aggregated_series + self.bottom_series
        self.reconciled_forecasts = pd.DataFrame(
            reconciled.T,
            index=self.forecasts.index,
            columns=all_series
        )
    
    def _reconcile_top_down(self):
        """Top-down reconciliation using historical proportions"""
        # Find the top-level series (root of hierarchy)
        all_children = set()
        for children in self.hierarchy.values():
            all_children.update(children)
        
        parents = set(self.hierarchy.keys())
        top_series_candidates = parents - all_children
        
        if len(top_series_candidates) == 0:
            raise ValueError("No top-level series found")
        
        top_series = list(top_series_candidates)[0]
        top_forecast = self.forecasts[top_series].values
        
        # Calculate proportions for each bottom series
        bottom_totals = self.forecasts[self.bottom_series].sum()
        total = bottom_totals.sum()
        proportions = (bottom_totals / total).values
        
        # Distribute top forecast proportionally
        reconciled_bottom = top_forecast[:, np.newaxis] * proportions
        
        # Calculate aggregated levels
        reconciled = self.summing_matrix @ reconciled_bottom.T
        
        all_series = self.aggregated_series + self.bottom_series
        self.reconciled_forecasts = pd.DataFrame(
            reconciled.T,
            index=self.forecasts.index,
            columns=all_series
        )
    
    def _reconcile_middle_out(self):
        """Middle-out reconciliation (simplified version)"""
        # Use bottom-up as default for middle-out
        # In practice, you'd specify a middle level
        self._reconcile_bottom_up()
    
    def _reconcile_mint_ols(self):
        """MinT optimal reconciliation with OLS"""
        # base forecasts vector for all series
        all_series = self.aggregated_series + self.bottom_series
        base_forecasts = self.forecasts[all_series].values
        
        n = base_forecasts.shape[0]  # time periods
        m = len(all_series)  # total number of series
        k = len(self.bottom_series)  # number of bottom series
        
        S = self.summing_matrix
        
        # OLS reconciliation: P = S(S'S)^{-1}S'
        # Reconciled = S @ (S'S)^{-1} @ S' @ base_forecasts
        
        try:
            # Compute projection matrix
            STS = S.T @ S
            STS_inv = np.linalg.inv(STS)
            P = S @ STS_inv @ S.T
            
            # Apply reconciliation
            reconciled = (P @ base_forecasts.T).T
            
            self.reconciled_forecasts = pd.DataFrame(
                reconciled,
                index=self.forecasts.index,
                columns=all_series
            )
        except np.linalg.LinAlgError:
            print("Warning: Matrix inversion failed, using bottom-up instead")
            self._reconcile_bottom_up()
    
    def get_reconciled_forecasts(self) -> pd.DataFrame:
        """Return reconciled forecasts
        
        Returns:
            DataFrame with reconciled forecasts
        """
        if self.reconciled_forecasts is None:
            raise ValueError("No reconciled forecasts. Run reconcile() first.")
        
        return self.reconciled_forecasts
    
    def validate_coherency(self, tolerance: float = 1e-6) -> bool:
        """Validate that reconciled forecasts satisfy hierarchical constraints
        
        Args:
            tolerance: Numerical tolerance for validation
            
        Returns:
            True if coherent, False otherwise
        """
        if self.reconciled_forecasts is None:
            raise ValueError("No reconciled forecasts. Run reconcile() first.")
        
        all_series = self.aggregated_series + self.bottom_series
        reconciled = self.reconciled_forecasts[all_series].values
        
        # Check if S @ bottom_reconciled ≈ all_reconciled
        bottom_reconciled = self.reconciled_forecasts[self.bottom_series].values
        expected = (self.summing_matrix @ bottom_reconciled.T).T
        
        max_error = np.max(np.abs(reconciled - expected))
        
        return max_error < tolerance
    
    def get_reconciliation_info(self) -> Dict:
        """Get information about the reconciliation
        
        Returns:
            Dict with hierarchy and reconciliation info
        """
        return {
            'n_bottom_series': len(self.bottom_series),
            'n_aggregated_series': len(self.aggregated_series),
            'n_total_series': len(self.bottom_series) + len(self.aggregated_series),
            'bottom_series': self.bottom_series,
            'aggregated_series': self.aggregated_series,
            'summing_matrix_shape': self.summing_matrix.shape if self.summing_matrix is not None else None,
            'is_coherent': self.validate_coherency() if self.reconciled_forecasts is not None else None
        }