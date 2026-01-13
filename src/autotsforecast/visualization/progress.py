"""
Rich progress tracking for time series forecasting operations.

Provides visual progress bars for model training, backtesting, and forecasting.
"""

import sys
from typing import Optional, Iterator, Any, Callable
from contextlib import contextmanager
import time


# Try to import rich for beautiful progress bars
try:
    from rich.progress import (
        Progress, 
        SpinnerColumn, 
        TextColumn, 
        BarColumn, 
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
        MofNCompleteColumn
    )
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Fallback to tqdm if rich not available
try:
    from tqdm import tqdm
    from tqdm.auto import tqdm as tqdm_auto
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProgressTracker:
    """
    Unified progress tracking for forecasting operations.
    
    Automatically selects the best available progress bar library:
    1. Rich (beautiful, feature-rich)
    2. tqdm (widely available)
    3. Simple print statements (fallback)
    
    Parameters
    ----------
    total : int
        Total number of iterations
    description : str, default=''
        Description of the task
    unit : str, default='it'
        Unit name for iterations
    disable : bool, default=False
        Disable progress bar entirely
    style : str, default='auto'
        Progress bar style: 'auto', 'rich', 'tqdm', 'simple'
        
    Examples
    --------
    >>> from autotsforecast.visualization import ProgressTracker
    >>> 
    >>> # As context manager
    >>> with ProgressTracker(total=100, description="Training") as pbar:
    ...     for i in range(100):
    ...         # do work
    ...         pbar.update(1)
    >>> 
    >>> # As iterator wrapper
    >>> for item in ProgressTracker.track(items, description="Processing"):
    ...     process(item)
    """
    
    def __init__(
        self,
        total: int,
        description: str = '',
        unit: str = 'it',
        disable: bool = False,
        style: str = 'auto'
    ):
        self.total = total
        self.description = description
        self.unit = unit
        self.disable = disable
        self.style = style
        
        self._progress = None
        self._task_id = None
        self._current = 0
        self._start_time = None
        
    def _select_backend(self):
        """Select the best available progress bar backend."""
        if self.style == 'rich' and RICH_AVAILABLE:
            return 'rich'
        elif self.style == 'tqdm' and TQDM_AVAILABLE:
            return 'tqdm'
        elif self.style == 'auto':
            if RICH_AVAILABLE:
                return 'rich'
            elif TQDM_AVAILABLE:
                return 'tqdm'
        return 'simple'
    
    def __enter__(self):
        if self.disable:
            return self
        
        self._start_time = time.time()
        backend = self._select_backend()
        
        if backend == 'rich':
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self.description, 
                total=self.total
            )
        elif backend == 'tqdm':
            self._progress = tqdm_auto(
                total=self.total,
                desc=self.description,
                unit=self.unit
            )
        else:
            print(f"Starting: {self.description} (0/{self.total})", end='', flush=True)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return
        
        backend = self._select_backend()
        
        if backend == 'rich' and self._progress:
            self._progress.stop()
        elif backend == 'tqdm' and self._progress:
            self._progress.close()
        else:
            elapsed = time.time() - self._start_time if self._start_time else 0
            print(f"\r✓ {self.description}: Done! ({elapsed:.1f}s)")
        
        return False
    
    def update(self, n: int = 1, **kwargs):
        """Update progress by n steps."""
        if self.disable:
            return
        
        self._current += n
        backend = self._select_backend()
        
        if backend == 'rich' and self._progress:
            self._progress.update(self._task_id, advance=n, **kwargs)
        elif backend == 'tqdm' and self._progress:
            self._progress.update(n)
        else:
            # Simple progress
            pct = self._current / self.total * 100 if self.total > 0 else 0
            print(f"\r{self.description}: {self._current}/{self.total} ({pct:.0f}%)", 
                  end='', flush=True)
    
    def set_description(self, description: str):
        """Update the task description."""
        self.description = description
        backend = self._select_backend()
        
        if backend == 'rich' and self._progress:
            self._progress.update(self._task_id, description=description)
        elif backend == 'tqdm' and self._progress:
            self._progress.set_description(description)
    
    @classmethod
    def track(
        cls,
        iterable,
        total: Optional[int] = None,
        description: str = '',
        disable: bool = False,
        **kwargs
    ) -> Iterator:
        """
        Track progress through an iterable.
        
        Parameters
        ----------
        iterable : iterable
            Items to iterate over
        total : int, optional
            Total count (if iterable doesn't have __len__)
        description : str
            Task description
        disable : bool
            Disable progress bar
            
        Yields
        ------
        Items from the iterable
        """
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        
        if disable:
            yield from iterable
            return
        
        if RICH_AVAILABLE:
            from rich.progress import track as rich_track
            yield from rich_track(
                iterable,
                total=total,
                description=description,
                **kwargs
            )
        elif TQDM_AVAILABLE:
            yield from tqdm_auto(
                iterable,
                total=total,
                desc=description,
                **kwargs
            )
        else:
            for i, item in enumerate(iterable):
                if total:
                    pct = (i + 1) / total * 100
                    print(f"\r{description}: {i+1}/{total} ({pct:.0f}%)", 
                          end='', flush=True)
                yield item
            print()


def progress_bar(
    iterable=None,
    total: Optional[int] = None,
    description: str = '',
    disable: bool = False,
    **kwargs
):
    """
    Convenience function to create a progress bar.
    
    Can be used as either:
    1. Iterator wrapper: for item in progress_bar(items): ...
    2. Context manager: with progress_bar(total=100) as pbar: pbar.update()
    
    Parameters
    ----------
    iterable : iterable, optional
        Items to iterate over
    total : int, optional
        Total number of items
    description : str
        Task description
    disable : bool
        Disable progress bar
        
    Returns
    -------
    ProgressTracker or iterator
    """
    if iterable is not None:
        return ProgressTracker.track(
            iterable, 
            total=total, 
            description=description, 
            disable=disable,
            **kwargs
        )
    elif total is not None:
        return ProgressTracker(
            total=total,
            description=description,
            disable=disable,
            **kwargs
        )
    else:
        raise ValueError("Either iterable or total must be provided")


class MultiProgressTracker:
    """
    Track multiple concurrent progress bars (e.g., per-series model training).
    
    Examples
    --------
    >>> tracker = MultiProgressTracker()
    >>> with tracker:
    ...     for series in series_list:
    ...         task_id = tracker.add_task(f"Training {series}", total=10)
    ...         for step in range(10):
    ...             # train
    ...             tracker.update(task_id)
    """
    
    def __init__(self, disable: bool = False):
        self.disable = disable
        self._progress = None
        self._tasks = {}
    
    def __enter__(self):
        if self.disable or not RICH_AVAILABLE:
            return self
        
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        self._progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._progress:
            self._progress.stop()
        return False
    
    def add_task(self, description: str, total: int) -> Any:
        """Add a new task to track."""
        if self._progress:
            task_id = self._progress.add_task(description, total=total)
            self._tasks[description] = task_id
            return task_id
        return description
    
    def update(self, task_id: Any, n: int = 1, **kwargs):
        """Update a task's progress."""
        if self._progress and task_id is not None:
            self._progress.update(task_id, advance=n, **kwargs)
    
    def complete_task(self, task_id: Any):
        """Mark a task as complete."""
        if self._progress and task_id is not None:
            task = self._progress._tasks.get(task_id)
            if task:
                remaining = task.total - task.completed
                self._progress.update(task_id, advance=remaining)


def print_banner(title: str, subtitle: str = ''):
    """Print a formatted banner."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel(
            f"[bold]{subtitle}[/bold]" if subtitle else "",
            title=f"[bold blue]{title}[/bold blue]",
            border_style="blue"
        ))
    else:
        print("=" * 60)
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print("=" * 60)


def print_results_table(results: dict, title: str = "Results"):
    """Print results in a formatted table."""
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title=title)
        
        # Add columns
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in results.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
        
        console.print(table)
    else:
        print(f"\n{title}")
        print("-" * 40)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
