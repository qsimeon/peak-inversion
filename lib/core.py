"""Core algorithm for finding global maxima using iterative peak finding.

This module implements an algorithm that finds global maxima by:
1. Finding local peaks using classical peak detection
2. Inverting the function around discovered peaks
3. Iterating with different seeds to explore the search space
4. Tracking and returning the best global maximum found
"""

from typing import Callable, List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class PeakResult:
    """Result of a peak finding operation.
    
    Attributes:
        position: The x-coordinate(s) of the peak
        value: The function value at the peak
        iteration: Which iteration this peak was found in
        seed: The seed value used to find this peak
    """
    position: np.ndarray
    value: float
    iteration: int
    seed: Optional[float] = None


class GlobalMaximaFinder:
    """Find global maxima using iterative peak finding with inversion.
    
    This class implements an algorithm that searches for global maxima by:
    - Starting from different seed points
    - Finding local peaks using gradient-based or derivative methods
    - Inverting the function around found peaks to escape local maxima
    - Iterating until convergence or max iterations reached
    """
    
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: Tuple[float, float],
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        num_seeds: int = 5
    ):
        """Initialize the global maxima finder.
        
        Args:
            func: The function to find maxima for. Should accept numpy array.
            bounds: Tuple of (lower_bound, upper_bound) for search space
            max_iterations: Maximum number of iterations per seed
            tolerance: Convergence tolerance for peak detection
            num_seeds: Number of different seeds to try
        """
        self.func = func
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.num_seeds = num_seeds
        self.peak_history: List[PeakResult] = []
    
    def find_local_peak(
        self,
        start_point: float,
        inverted: bool = False
    ) -> Tuple[float, float]:
        """Find a local peak starting from a given point.
        
        Uses a simple gradient ascent approach to find local maxima.
        
        Args:
            start_point: Starting position for peak search
            inverted: Whether to search on inverted function
        
        Returns:
            Tuple of (position, value) of the found peak
        """
        current_pos = np.clip(start_point, self.bounds[0], self.bounds[1])
        step_size = (self.bounds[1] - self.bounds[0]) * 0.01
        
        for _ in range(100):  # Local search iterations
            # Compute numerical gradient
            epsilon = 1e-8
            grad = self._numerical_gradient(current_pos, epsilon, inverted)
            
            # Update position
            new_pos = current_pos + step_size * grad
            new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
            
            # Check convergence
            if abs(new_pos - current_pos) < self.tolerance:
                break
            
            current_pos = new_pos
            step_size *= 0.95  # Adaptive step size
        
        value = self.func(np.array([current_pos]))
        if inverted:
            value = -value
        
        return float(current_pos), float(value)
    
    def _numerical_gradient(
        self,
        x: float,
        epsilon: float,
        inverted: bool
    ) -> float:
        """Compute numerical gradient at point x.
        
        Args:
            x: Point to compute gradient at
            epsilon: Small value for numerical differentiation
            inverted: Whether function is inverted
        
        Returns:
            Gradient value
        """
        x_plus = np.clip(x + epsilon, self.bounds[0], self.bounds[1])
        x_minus = np.clip(x - epsilon, self.bounds[0], self.bounds[1])
        
        f_plus = self.func(np.array([x_plus]))
        f_minus = self.func(np.array([x_minus]))
        
        if inverted:
            f_plus = -f_plus
            f_minus = -f_minus
        
        grad = (f_plus - f_minus) / (x_plus - x_minus)
        return float(grad)
    
    def _invert_around_peak(
        self,
        peak_pos: float,
        x: float
    ) -> float:
        """Create an inverted version of the function around a peak.
        
        Args:
            peak_pos: Position of the peak to invert around
            x: Point to evaluate
        
        Returns:
            Modified x value that explores inverted region
        """
        # Reflect point around peak and add perturbation
        distance = x - peak_pos
        inverted_x = peak_pos - distance * 0.5
        return np.clip(inverted_x, self.bounds[0], self.bounds[1])
    
    def find_global_maximum(self) -> PeakResult:
        """Find the global maximum using iterative peak finding.
        
        Returns:
            PeakResult containing the best global maximum found
        """
        best_peak: Optional[PeakResult] = None
        
        # Generate seed points
        seeds = np.linspace(
            self.bounds[0],
            self.bounds[1],
            self.num_seeds
        )
        
        for seed_idx, seed in enumerate(seeds):
            current_seed = float(seed)
            
            for iteration in range(self.max_iterations):
                # Find local peak from current seed
                peak_pos, peak_val = self.find_local_peak(current_seed)
                
                # Create peak result
                peak_result = PeakResult(
                    position=np.array([peak_pos]),
                    value=peak_val,
                    iteration=iteration,
                    seed=float(seed)
                )
                
                self.peak_history.append(peak_result)
                
                # Update best peak if this is better
                if best_peak is None or peak_val > best_peak.value:
                    best_peak = peak_result
                
                # Generate new seed by inverting around found peak
                # Add some randomness to explore different regions
                offset = (self.bounds[1] - self.bounds[0]) * 0.1 * (0.5 - np.random.random())
                current_seed = self._invert_around_peak(peak_pos, peak_pos + offset)
                
                # Try a completely new region occasionally
                if iteration % 3 == 2:
                    current_seed = np.random.uniform(self.bounds[0], self.bounds[1])
        
        return best_peak if best_peak is not None else PeakResult(
            position=np.array([self.bounds[0]]),
            value=float('-inf'),
            iteration=0
        )
    
    def get_peak_history(self) -> List[PeakResult]:
        """Get the history of all peaks found during search.
        
        Returns:
            List of all PeakResult objects found
        """
        return self.peak_history.copy()
    
    def reset(self) -> None:
        """Reset the peak history and internal state."""
        self.peak_history.clear()


def find_peaks_1d(
    func: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    **kwargs
) -> PeakResult:
    """Convenience function to find global maximum in 1D.
    
    Args:
        func: Function to find maximum of
        bounds: Search bounds as (lower, upper)
        **kwargs: Additional arguments passed to GlobalMaximaFinder
    
    Returns:
        PeakResult with the global maximum found
    
    Example:
        >>> def f(x): return -(x[0] - 2)**2 + 5
        >>> result = find_peaks_1d(f, (0, 10))
        >>> print(f"Maximum at x={result.position[0]:.3f}, value={result.value:.3f}")
    """
    finder = GlobalMaximaFinder(func, bounds, **kwargs)
    return finder.find_global_maximum()
