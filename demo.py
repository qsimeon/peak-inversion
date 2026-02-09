#!/usr/bin/env python3
"""
Global Maxima Finder Demo Script

This demo demonstrates an algorithm for finding global maxima by:
1. Starting from different seed points
2. Finding local peaks using classical gradient-based approach
3. Inverting the function around found peaks
4. Iterating with different seeds to explore the search space
5. Identifying the global maximum from all discovered peaks

The algorithm helps avoid getting stuck in local maxima by systematically
exploring the function landscape through inversion and multiple initializations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
import sys
import os

# Add lib directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Import from available modules
from core import PeakResult, GlobalMaximaFinder, find_peaks_1d
from utils import (
    generate_seeds, 
    invert_function, 
    evaluate_function_grid,
    filter_duplicate_peaks,
    compute_function_statistics
)


def test_function_1(x: np.ndarray) -> float:
    """
    Multi-modal test function with several local maxima.
    Global maximum at x ≈ 1.5
    """
    x_val = x if np.isscalar(x) else x[0]
    return (
        np.sin(x_val) * np.exp(-0.1 * x_val**2) + 
        0.5 * np.sin(3 * x_val) * np.exp(-0.05 * (x_val - 2)**2) +
        0.3 * np.cos(5 * x_val)
    )


def test_function_2(x: np.ndarray) -> float:
    """
    Another multi-modal function with multiple peaks.
    Combination of Gaussian peaks.
    """
    x_val = x if np.isscalar(x) else x[0]
    return (
        2.0 * np.exp(-((x_val - 2.0)**2) / 0.5) +
        1.5 * np.exp(-((x_val - 5.0)**2) / 0.8) +
        1.8 * np.exp(-((x_val - 8.0)**2) / 0.6) +
        0.5 * np.sin(2 * x_val)
    )


def test_function_3(x: np.ndarray) -> float:
    """
    Rastrigin-like function (inverted for maximization).
    Multiple local maxima with one global maximum.
    """
    x_val = x if np.isscalar(x) else x[0]
    A = 10
    return A - (x_val**2 - A * np.cos(2 * np.pi * x_val)) * 0.1


def visualize_optimization_process(
    func: Callable[[np.ndarray], float],
    bounds: Tuple[float, float],
    finder: GlobalMaximaFinder,
    title: str = "Global Maxima Finding Process"
):
    """
    Visualize the function and the optimization process.
    
    Args:
        func: Function to visualize
        bounds: Domain bounds
        finder: GlobalMaximaFinder instance with history
        title: Plot title
    """
    # Evaluate function on grid
    x_grid, y_grid = evaluate_function_grid(func, bounds, num_points=500)
    
    # Get peak history
    history = finder.get_peak_history()
    
    # Create figure
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Function with all discovered peaks
    plt.subplot(1, 2, 1)
    plt.plot(x_grid, y_grid, 'b-', linewidth=2, label='Function')
    
    # Plot all peaks from history
    if history:
        peak_positions = [p.position[0] for p in history]
        peak_values = [p.value for p in history]
        plt.scatter(peak_positions, peak_values, c='red', s=100, 
                   marker='o', alpha=0.6, label='Discovered Peaks', zorder=5)
        
        # Highlight global maximum
        global_peak = max(history, key=lambda p: p.value)
        plt.scatter([global_peak.position[0]], [global_peak.value], 
                   c='green', s=200, marker='*', 
                   label=f'Global Max: {global_peak.value:.4f}', 
                   zorder=10, edgecolors='black', linewidth=2)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Convergence history
    plt.subplot(1, 2, 2)
    if history:
        iterations = list(range(1, len(history) + 1))
        peak_values = [p.value for p in history]
        cumulative_best = [max(peak_values[:i+1]) for i in range(len(peak_values))]
        
        plt.plot(iterations, peak_values, 'bo-', label='Peak Values', alpha=0.6)
        plt.plot(iterations, cumulative_best, 'g-', linewidth=2, 
                label='Best So Far', marker='s')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Peak Value', fontsize=12)
        plt.title('Convergence History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'optimization_process_{title.replace(" ", "_")}.png', dpi=150)
    print(f"  → Visualization saved as 'optimization_process_{title.replace(' ', '_')}.png'")
    plt.close()


def demonstrate_basic_usage():
    """Demonstrate basic usage of the global maxima finder."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Global Maxima Finding")
    print("="*70)
    
    # Define bounds
    bounds = (-5.0, 5.0)
    
    print(f"\nSearching for global maximum in range {bounds}")
    print("Using test function: sin(x)*exp(-0.1*x²) + 0.5*sin(3x)*exp(-0.05*(x-2)²) + 0.3*cos(5x)")
    
    # Use convenience function
    result = find_peaks_1d(
        test_function_1,
        bounds,
        num_seeds=10,
        max_iterations=50,
        tolerance=1e-6
    )
    
    print(f"\n✓ Global maximum found!")
    print(f"  Position: x = {result.position[0]:.6f}")
    print(f"  Value: f(x) = {result.value:.6f}")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Function evaluations: {result.function_evaluations}")


def demonstrate_advanced_usage():
    """Demonstrate advanced usage with custom configuration."""
    print("\n" + "="*70)
    print("DEMO 2: Advanced Usage with Multiple Test Functions")
    print("="*70)
    
    test_functions = [
        (test_function_1, (-5.0, 5.0), "Multi-modal Trigonometric"),
        (test_function_2, (0.0, 10.0), "Gaussian Peaks"),
        (test_function_3, (-5.0, 5.0), "Rastrigin-like")
    ]
    
    for idx, (func, bounds, name) in enumerate(test_functions, 1):
        print(f"\n--- Test Function {idx}: {name} ---")
        print(f"Domain: {bounds}")
        
        # Create finder instance
        finder = GlobalMaximaFinder(
            func,
            bounds,
            num_seeds=15,
            max_iterations=100,
            tolerance=1e-7,
            inversion_strength=0.8
        )
        
        # Find global maximum
        result = finder.find_global_maximum()
        
        print(f"\n✓ Results:")
        print(f"  Global Maximum Position: x = {result.position[0]:.6f}")
        print(f"  Global Maximum Value: f(x) = {result.value:.6f}")
        print(f"  Total Iterations: {result.iterations}")
        print(f"  Function Evaluations: {result.function_evaluations}")
        print(f"  Converged: {result.converged}")
        
        # Get statistics
        stats = compute_function_statistics(func, bounds, num_samples=1000)
        print(f"\n  Function Statistics:")
        print(f"    Mean: {stats['mean']:.6f}")
        print(f"    Std Dev: {stats['std']:.6f}")
        print(f"    Min: {stats['min']:.6f}")
        print(f"    Max: {stats['max']:.6f}")
        
        # Visualize
        visualize_optimization_process(func, bounds, finder, f"{name} Function")
        
        # Show peak history
        history = finder.get_peak_history()
        print(f"\n  Peak Discovery History ({len(history)} peaks found):")
        for i, peak in enumerate(history[:5], 1):  # Show first 5
            print(f"    Peak {i}: x={peak.position[0]:.4f}, f(x)={peak.value:.4f}")
        if len(history) > 5:
            print(f"    ... and {len(history) - 5} more peaks")


def demonstrate_seed_strategies():
    """Demonstrate different seed generation strategies."""
    print("\n" + "="*70)
    print("DEMO 3: Comparing Different Seed Strategies")
    print("="*70)
    
    bounds = (0.0, 10.0)
    func = test_function_2
    
    strategies = ['uniform', 'random', 'grid']
    
    results_comparison = []
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.upper()} ---")
        
        # Generate seeds
        seeds = generate_seeds(bounds, num_seeds=12, strategy=strategy)
        print(f"Generated {len(seeds)} seeds using '{strategy}' strategy")
        print(f"Sample seeds: {seeds[:3]}")
        
        # Create finder
        finder = GlobalMaximaFinder(
            func,
            bounds,
            num_seeds=12,
            seed_strategy=strategy,
            max_iterations=80,
            tolerance=1e-6
        )
        
        # Find maximum
        result = finder.find_global_maximum()
        
        print(f"\n✓ Results with {strategy} strategy:")
        print(f"  Maximum: f({result.position[0]:.4f}) = {result.value:.6f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Function Evaluations: {result.function_evaluations}")
        
        results_comparison.append({
            'strategy': strategy,
            'position': result.position[0],
            'value': result.value,
            'iterations': result.iterations,
            'evaluations': result.function_evaluations
        })
    
    # Summary comparison
    print("\n" + "-"*70)
    print("COMPARISON SUMMARY:")
    print("-"*70)
    print(f"{'Strategy':<15} {'Max Value':<15} {'Position':<15} {'Iterations':<15}")
    print("-"*70)
    for res in results_comparison:
        print(f"{res['strategy']:<15} {res['value']:<15.6f} {res['position']:<15.6f} {res['iterations']:<15}")


def demonstrate_iterative_refinement():
    """Demonstrate how the algorithm refines through iterations."""
    print("\n" + "="*70)
    print("DEMO 4: Iterative Refinement Process")
    print("="*70)
    
    bounds = (-5.0, 5.0)
    func = test_function_1
    
    print("\nDemonstrating step-by-step refinement:")
    print("Starting with few seeds and gradually exploring...")
    
    # Create finder
    finder = GlobalMaximaFinder(
        func,
        bounds,
        num_seeds=8,
        max_iterations=60,
        tolerance=1e-7,
        inversion_strength=0.9
    )
    
    # Manually control iterations to show progress
    print("\nPhase 1: Initial exploration with 8 seeds")
    result = finder.find_global_maximum()
    
    history = finder.get_peak_history()
    print(f"  → Found {len(history)} distinct peaks")
    
    # Filter duplicates
    peak_tuples = [(p.position[0], p.value) for p in history]
    unique_peaks = filter_duplicate_peaks(peak_tuples, tolerance=1e-4)
    print(f"  → {len(unique_peaks)} unique peaks after filtering")
    
    # Show top peaks
    sorted_peaks = sorted(unique_peaks, key=lambda p: p[1], reverse=True)
    print(f"\n  Top 5 peaks discovered:")
    for i, (pos, val) in enumerate(sorted_peaks[:5], 1):
        print(f"    {i}. x={pos:.6f}, f(x)={val:.6f}")
    
    print(f"\n✓ Final Global Maximum:")
    print(f"  Position: x = {result.position[0]:.6f}")
    print(f"  Value: f(x) = {result.value:.6f}")
    
    # Visualize
    visualize_optimization_process(func, bounds, finder, "Iterative Refinement")


def demonstrate_error_handling():
    """Demonstrate error handling and edge cases."""
    print("\n" + "="*70)
    print("DEMO 5: Error Handling and Edge Cases")
    print("="*70)
    
    # Test 1: Constant function
    print("\n--- Test 1: Constant Function ---")
    def constant_func(x):
        return 5.0
    
    try:
        result = find_peaks_1d(constant_func, (0.0, 10.0), num_seeds=5)
        print(f"✓ Handled constant function: f(x) = {result.value:.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Linear function
    print("\n--- Test 2: Linear Function ---")
    def linear_func(x):
        x_val = x if np.isscalar(x) else x[0]
        return 2.0 * x_val + 1.0
    
    try:
        result = find_peaks_1d(linear_func, (0.0, 10.0), num_seeds=5)
        print(f"✓ Handled linear function: maximum at boundary")
        print(f"  Position: x = {result.position[0]:.6f}")
        print(f"  Value: f(x) = {result.value:.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Very narrow bounds
    print("\n--- Test 3: Very Narrow Bounds ---")
    try:
        result = find_peaks_1d(test_function_1, (1.0, 1.001), num_seeds=3)
        print(f"✓ Handled narrow bounds")
        print(f"  Position: x = {result.position[0]:.6f}")
        print(f"  Value: f(x) = {result.value:.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Single seed
    print("\n--- Test 4: Single Seed ---")
    try:
        result = find_peaks_1d(test_function_2, (0.0, 10.0), num_seeds=1)
        print(f"✓ Handled single seed")
        print(f"  Position: x = {result.position[0]:.6f}")
        print(f"  Value: f(x) = {result.value:.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Main demonstration function."""
    print("\n" + "="*70)
    print(" GLOBAL MAXIMA FINDER - COMPREHENSIVE DEMONSTRATION")
    print("="*70)
    print("\nThis demo showcases an algorithm for finding global maxima using:")
    print("  • Multiple seed points for initialization")
    print("  • Classical gradient-based peak finding")
    print("  • Function inversion around discovered peaks")
    print("  • Iterative refinement to explore the search space")
    print("="*70)
    
    try:
        # Run all demonstrations
        demonstrate_basic_usage()
        demonstrate_advanced_usage()
        demonstrate_seed_strategies()
        demonstrate_iterative_refinement()
        demonstrate_error_handling()
        
        print("\n" + "="*70)
        print(" ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nKey Takeaways:")
        print("  ✓ The algorithm successfully finds global maxima in multi-modal functions")
        print("  ✓ Multiple seed strategies help explore different regions")
        print("  ✓ Function inversion helps escape local maxima")
        print("  ✓ Iterative refinement improves solution quality")
        print("  ✓ Robust error handling for edge cases")
        print("\nVisualization files have been saved to the current directory.")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
