"""Utility functions for peak finding and function analysis.

This module provides helper functions for:
- Function evaluation and transformation
- Numerical analysis (derivatives, gradients)
- Seed generation strategies
- Result visualization and analysis
"""

from typing import Callable, List, Tuple, Optional, Union
import numpy as np


def generate_seeds(
    bounds: Tuple[float, float],
    num_seeds: int,
    strategy: str = 'uniform'
) -> np.ndarray:
    """Generate seed points for peak finding initialization.
    
    Args:
        bounds: Tuple of (lower_bound, upper_bound)
        num_seeds: Number of seed points to generate
        strategy: Strategy for seed generation:
            - 'uniform': Evenly spaced points
            - 'random': Randomly distributed points
            - 'clustered': Clustered around potential regions
    
    Returns:
        Array of seed points
    
    Raises:
        ValueError: If strategy is not recognized
    """
    lower, upper = bounds
    
    if strategy == 'uniform':
        return np.linspace(lower, upper, num_seeds)
    elif strategy == 'random':
        return np.random.uniform(lower, upper, num_seeds)
    elif strategy == 'clustered':
        # Generate clusters around quartiles
        centers = np.linspace(lower, upper, 4)
        seeds = []
        seeds_per_cluster = num_seeds // 4
        spread = (upper - lower) * 0.1
        
        for center in centers:
            cluster = np.random.normal(center, spread, seeds_per_cluster)
            cluster = np.clip(cluster, lower, upper)
            seeds.extend(cluster)
        
        return np.array(seeds[:num_seeds])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def invert_function(
    func: Callable[[np.ndarray], float],
    peak_position: float,
    inversion_strength: float = 1.0
) -> Callable[[np.ndarray], float]:
    """Create an inverted version of a function around a peak.
    
    Args:
        func: Original function
        peak_position: Position to invert around
        inversion_strength: Strength of inversion (0 to 1)
    
    Returns:
        Inverted function
    """
    def inverted_func(x: np.ndarray) -> float:
        # Reflect input around peak
        distance = x - peak_position
        reflected_x = peak_position - distance * inversion_strength
        
        # Evaluate original function at reflected point
        return func(reflected_x)
    
    return inverted_func


def numerical_derivative(
    func: Callable[[float], float],
    x: float,
    h: float = 1e-8,
    method: str = 'central'
) -> float:
    """Compute numerical derivative of a function.
    
    Args:
        func: Function to differentiate
        x: Point to compute derivative at
        h: Step size for numerical differentiation
        method: Method to use:
            - 'forward': Forward difference
            - 'backward': Backward difference
            - 'central': Central difference (more accurate)
    
    Returns:
        Approximate derivative value
    """
    if method == 'forward':
        return (func(x + h) - func(x)) / h
    elif method == 'backward':
        return (func(x) - func(x - h)) / h
    elif method == 'central':
        return (func(x + h) - func(x - h)) / (2 * h)
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_local_maxima(
    values: np.ndarray,
    threshold: Optional[float] = None
) -> np.ndarray:
    """Detect local maxima in a 1D array of values.
    
    Args:
        values: Array of function values
        threshold: Optional minimum value for peaks
    
    Returns:
        Array of indices where local maxima occur
    """
    if len(values) < 3:
        return np.array([])
    
    # Find points where derivative changes from positive to negative
    maxima_indices = []
    
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            if threshold is None or values[i] >= threshold:
                maxima_indices.append(i)
    
    return np.array(maxima_indices)


def evaluate_function_grid(
    func: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate function on a regular grid.
    
    Useful for visualization and analysis.
    
    Args:
        func: Function to evaluate
        bounds: Tuple of (lower_bound, upper_bound)
        num_points: Number of grid points
    
    Returns:
        Tuple of (x_values, y_values)
    """
    x_values = np.linspace(bounds[0], bounds[1], num_points)
    y_values = np.array([func(np.array([x])) for x in x_values])
    
    return x_values, y_values


def compute_function_statistics(
    func: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    num_samples: int = 1000
) -> dict:
    """Compute statistical properties of a function over a domain.
    
    Args:
        func: Function to analyze
        bounds: Domain bounds
        num_samples: Number of sample points
    
    Returns:
        Dictionary with statistics: mean, std, min, max, range
    """
    x_samples = np.random.uniform(bounds[0], bounds[1], num_samples)
    y_samples = np.array([func(np.array([x])) for x in x_samples])
    
    return {
        'mean': float(np.mean(y_samples)),
        'std': float(np.std(y_samples)),
        'min': float(np.min(y_samples)),
        'max': float(np.max(y_samples)),
        'range': float(np.max(y_samples) - np.min(y_samples)),
        'median': float(np.median(y_samples))
    }


def normalize_function(
    func: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    target_range: Tuple[float, float] = (0.0, 1.0),
    num_samples: int = 100
) -> Callable[[np.ndarray], float]:
    """Create a normalized version of a function.
    
    Args:
        func: Original function
        bounds: Domain bounds for sampling
        target_range: Desired output range (min, max)
        num_samples: Number of samples for estimating range
    
    Returns:
        Normalized function
    """
    # Estimate function range
    x_samples = np.linspace(bounds[0], bounds[1], num_samples)
    y_samples = np.array([func(np.array([x])) for x in x_samples])
    
    y_min = np.min(y_samples)
    y_max = np.max(y_samples)
    y_range = y_max - y_min
    
    if y_range == 0:
        y_range = 1.0  # Avoid division by zero
    
    target_min, target_max = target_range
    target_span = target_max - target_min
    
    def normalized_func(x: np.ndarray) -> float:
        y = func(x)
        # Normalize to [0, 1] then scale to target range
        normalized = (y - y_min) / y_range
        return target_min + normalized * target_span
    
    return normalized_func


def compare_peaks(
    peak1: Tuple[float, float],
    peak2: Tuple[float, float],
    position_tolerance: float = 1e-6,
    value_tolerance: float = 1e-6
) -> bool:
    """Compare two peaks for equality within tolerance.
    
    Args:
        peak1: First peak as (position, value)
        peak2: Second peak as (position, value)
        position_tolerance: Tolerance for position comparison
        value_tolerance: Tolerance for value comparison
    
    Returns:
        True if peaks are considered equal
    """
    pos_diff = abs(peak1[0] - peak2[0])
    val_diff = abs(peak1[1] - peak2[1])
    
    return pos_diff <= position_tolerance and val_diff <= value_tolerance


def filter_duplicate_peaks(
    peaks: List[Tuple[float, float]],
    tolerance: float = 1e-6
) -> List[Tuple[float, float]]:
    """Remove duplicate peaks from a list.
    
    Args:
        peaks: List of (position, value) tuples
        tolerance: Tolerance for considering peaks as duplicates
    
    Returns:
        List of unique peaks
    """
    if not peaks:
        return []
    
    unique_peaks = [peaks[0]]
    
    for peak in peaks[1:]:
        is_duplicate = False
        for unique_peak in unique_peaks:
            if compare_peaks(peak, unique_peak, tolerance, tolerance):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_peaks.append(peak)
    
    return unique_peaks


def adaptive_step_size(
    gradient: float,
    initial_step: float,
    min_step: float = 1e-10,
    max_step: float = 1.0
) -> float:
    """Compute adaptive step size based on gradient magnitude.
    
    Args:
        gradient: Current gradient value
        initial_step: Initial step size
        min_step: Minimum allowed step size
        max_step: Maximum allowed step size
    
    Returns:
        Adapted step size
    """
    # Reduce step size for large gradients, increase for small gradients
    grad_magnitude = abs(gradient)
    
    if grad_magnitude > 1.0:
        step = initial_step / grad_magnitude
    elif grad_magnitude < 0.1 and grad_magnitude > 0:
        step = initial_step * 2.0
    else:
        step = initial_step
    
    return np.clip(step, min_step, max_step)
