"""
Lightweight local model registry for AutoTSForecast.

Stores fitted AutoForecaster instances with metadata so agents can
fit once and retrieve later without re-training.

Default registry path: ``~/.autotsforecast/registry/``

Usage::

    from autotsforecast.registry.store import ModelRegistry

    registry = ModelRegistry()

    # Save a fitted model
    registry.save(auto, name="sales_forecast_v1", tags={"product": "A", "horizon": 30})

    # List all registered models
    print(registry.list())

    # Load a model and forecast immediately
    model = registry.load("sales_forecast_v1")
    forecasts = model.forecast()

    # Delete
    registry.delete("sales_forecast_v1")
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class ModelRegistry:
    """
    Lightweight local model registry backed by joblib + JSON index.

    Parameters
    ----------
    registry_dir : str or Path, optional
        Directory to store models. Defaults to ``~/.autotsforecast/registry/``.
    """

    _INDEX_FILE = "registry_index.json"

    def __init__(self, registry_dir: Optional[str] = None):
        if registry_dir is None:
            registry_dir = Path.home() / ".autotsforecast" / "registry"
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / self._INDEX_FILE
        self._index: Dict[str, dict] = self._load_index()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(
        self,
        model,
        name: str,
        tags: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Persist a fitted model to the registry.

        Parameters
        ----------
        model : AutoForecaster or any BaseForecaster
            A fitted forecasting model.
        name : str
            Unique name to identify this model in the registry.
        tags : dict, optional
            Arbitrary metadata (e.g. ``{"product": "A", "region": "US"}``).

        Returns
        -------
        str
            Registry entry ID (same as ``name``).
        """
        import joblib

        safe_name = _safe_filename(name)
        filepath = self.registry_dir / f"{safe_name}.joblib"

        # Collect metadata
        class_name = type(model).__name__
        horizon = getattr(model, "horizon", None) or getattr(model, "_horizon", None)
        metric = getattr(model, "metric", None)
        metric_value = getattr(model, "best_score_", None)

        metadata = {
            "model": model,
            "class_name": class_name,
            "horizon": horizon,
            "metric": metric,
            "metric_value": metric_value,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "tags": tags or {},
        }

        joblib.dump(metadata, filepath)

        entry = {
            "name": name,
            "class_name": class_name,
            "horizon": horizon,
            "metric": metric,
            "metric_value": metric_value,
            "saved_at": metadata["saved_at"],
            "tags": tags or {},
            "filepath": str(filepath),
        }
        self._index[name] = entry
        self._save_index()

        print(f"✓ Model '{name}' saved to registry  [{filepath}]")
        return name

    def load(self, name: str):
        """
        Load a fitted model from the registry.

        Parameters
        ----------
        name : str
            Registry entry name.

        Returns
        -------
        AutoForecaster or BaseForecaster
            The fitted model, ready to call ``.forecast()`` or ``.predict()``.

        Raises
        ------
        KeyError
            If ``name`` is not found in the registry.
        FileNotFoundError
            If the serialised file is missing.
        """
        import joblib

        if name not in self._index:
            available = list(self._index.keys())
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available: {available}"
            )

        entry = self._index[name]
        filepath = Path(entry["filepath"])
        if not filepath.exists():
            raise FileNotFoundError(
                f"Registry file not found: {filepath}. "
                "The model may have been moved or deleted."
            )

        metadata = joblib.load(filepath)
        model = metadata["model"]

        print(f"✓ Model '{name}' loaded from registry")
        print(f"  Class   : {entry['class_name']}")
        print(f"  Horizon : {entry['horizon']}")
        print(f"  Saved   : {entry['saved_at']}")
        if entry.get("tags"):
            print(f"  Tags    : {entry['tags']}")

        return model

    def delete(self, name: str) -> None:
        """
        Remove a model from the registry.

        Parameters
        ----------
        name : str
            Registry entry name to delete.
        """
        if name not in self._index:
            raise KeyError(f"Model '{name}' not found in registry.")

        entry = self._index.pop(name)
        filepath = Path(entry["filepath"])
        if filepath.exists():
            filepath.unlink()
        self._save_index()
        print(f"✓ Model '{name}' deleted from registry.")

    def list(self) -> pd.DataFrame:
        """
        List all registered models.

        Returns
        -------
        pd.DataFrame
            Columns: name, class_name, horizon, metric, metric_value, saved_at, tags.
        """
        if not self._index:
            return pd.DataFrame(columns=[
                "name", "class_name", "horizon", "metric",
                "metric_value", "saved_at", "tags",
            ])

        rows = []
        for entry in self._index.values():
            rows.append({
                "name": entry["name"],
                "class_name": entry["class_name"],
                "horizon": entry["horizon"],
                "metric": entry.get("metric"),
                "metric_value": entry.get("metric_value"),
                "saved_at": entry["saved_at"],
                "tags": entry.get("tags", {}),
            })
        return pd.DataFrame(rows)

    def get_entry(self, name: str):
        """
        Return structured registry metadata for a model.

        Returns
        -------
        RegistryEntry
            Pydantic model.
        """
        from autotsforecast.schemas import RegistryEntry

        if name not in self._index:
            raise KeyError(f"Model '{name}' not found.")
        entry = self._index[name]
        return RegistryEntry(**{k: v for k, v in entry.items()})

    def clear(self) -> None:
        """Remove all models from the registry (use with care)."""
        for name in list(self._index.keys()):
            self.delete(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, dict]:
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)


def _safe_filename(name: str) -> str:
    """Convert a registry name to a safe filesystem filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
