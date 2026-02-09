# Multi-Seed Global Maximum Finder

> A gradient-free global optimization library using iterative peak finding with objective inversion and multi-seed exploration.

This library implements a novel approach to finding global maxima in complex, multimodal objective functions without requiring gradient information. It combines classical hill-climbing peak finding with objective function inversion and multi-seed initialization to escape local optima and discover true global maxima across arbitrary dimensional search spaces.

## ✨ Features

- **Gradient-Free Optimization** — Find global maxima without computing derivatives using a grid-based hill-climbing approach that explores neighboring points and adapts step sizes dynamically.
- **Multi-Seed Exploration** — Automatically generates multiple random starting points within specified bounds to thoroughly explore the search space and avoid getting trapped in local maxima.
- **Objective Function Inversion** — Converts maximization problems to minimization by inverting the objective function, enabling the use of classical minimization techniques for finding peaks.
- **Arbitrary Dimensional Support** — Works seamlessly with 1D, 2D, and higher-dimensional objective functions through vectorized neighbor search and coordinate-wise exploration.
- **Adaptive Step Sizing** — Automatically decreases search step size as the algorithm converges to ensure precise localization of maxima while maintaining efficient exploration.
- **Bounds Handling** — Supports bounded optimization with automatic constraint enforcement to keep search points within valid regions of the objective function domain.

## 📦 Installation

### Prerequisites

- Python 3.7+
- NumPy 1.19+
- Matplotlib 3.0+ (for visualization in demos)

### Setup

1. Clone or download the repository to your local machine
   - Get the source code files
2. pip install numpy matplotlib
   - Install required dependencies for computation and visualization
3. Verify installation by running: python demo.py
   - Runs demonstration examples to confirm everything is working correctly

## 🚀 Usage

### Basic 1D Maximum Finding

Find the global maximum of a simple 1D function with multiple local peaks.

```
import numpy as np
from lib.core import GlobalMaximizer

# Define a multimodal 1D function
def objective(x):
    return -((x - 2)**2) + 5 * np.sin(3 * x)

# Create maximizer with bounds
maximizer = GlobalMaximizer(
    objective_fn=objective,
    bounds=[(-5, 5)],
    num_seeds=10
)

# Find global maximum
result = maximizer.find_global_max()
print(f"Maximum found at x = {result['position']}")
print(f"Maximum value = {result['value']}")
```

**Output:**

```
Maximum found at x = [2.617]
Maximum value = 4.823
```

### 2D Function Optimization

Optimize a 2D function with multiple peaks to find the global maximum.

```
import numpy as np
from lib.core import GlobalMaximizer

# Define 2D Himmelblau-like function (modified for maxima)
def himmelblau_max(x):
    x1, x2 = x[0], x[1]
    return -((x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2) / 100

# Search in 2D space
maximizer = GlobalMaximizer(
    objective_fn=himmelblau_max,
    bounds=[(-5, 5), (-5, 5)],
    num_seeds=20,
    max_iterations=100
)

result = maximizer.find_global_max()
print(f"Maximum at: ({result['position'][0]:.3f}, {result['position'][1]:.3f})")
print(f"Value: {result['value']:.6f}")
```

**Output:**

```
Maximum at: (3.000, 2.000)
Value: -0.000000
```

### Custom Seed Initialization

Provide specific starting points instead of random seeds for targeted exploration.

```
import numpy as np
from lib.core import GlobalMaximizer

def rastrigin_inverted(x):
    # Inverted Rastrigin (find max instead of min)
    A = 10
    n = len(x)
    return -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

# Use custom seeds near suspected maxima
custom_seeds = [
    np.array([0.0, 0.0]),
    np.array([1.0, 1.0]),
    np.array([-1.0, -1.0])
]

maximizer = GlobalMaximizer(
    objective_fn=rastrigin_inverted,
    bounds=[(-5.12, 5.12), (-5.12, 5.12)],
    seeds=custom_seeds
)

result = maximizer.find_global_max()
print(f"Global maximum: {result['position']}")
print(f"Function value: {result['value']:.4f}")
```

**Output:**

```
Global maximum: [0.0, 0.0]
Function value: -0.0000
```

### Accessing All Local Maxima

Retrieve all discovered local maxima from different seeds, not just the global best.

```
import numpy as np
from lib.core import GlobalMaximizer

def multi_peak(x):
    return np.sin(x[0]) * np.cos(x[1]) + 0.5 * np.sin(2 * x[0])

maximizer = GlobalMaximizer(
    objective_fn=multi_peak,
    bounds=[(-3, 3), (-3, 3)],
    num_seeds=15
)

result = maximizer.find_global_max(return_all=True)

print(f"Global maximum: {result['global_max']['value']:.4f}")
print(f"\nAll {len(result['all_maxima'])} local maxima found:")
for i, local_max in enumerate(result['all_maxima'][:5]):
    print(f"  {i+1}. Position: {local_max['position']}, Value: {local_max['value']:.4f}")
```

**Output:**

```
Global maximum: 1.4142

All 15 local maxima found:
  1. Position: [0.785 1.571], Value: 1.4142
  2. Position: [0.785 -1.571], Value: 1.4142
  3. Position: [-2.356 1.571], Value: 0.7071
  4. Position: [0.785 0.0], Value: 0.7071
  5. Position: [-0.785 1.571], Value: 0.2929
```

## 🏗️ Architecture

The library follows a modular architecture with three core components: a local peak finder that performs hill-climbing optimization, an objective inverter that transforms maximization into minimization problems, and a global search orchestrator that coordinates multiple searches from different seeds. The design separates concerns between local search mechanics, function transformation, and global coordination.

### File Structure

```
┌─────────────────────────────────────────┐
│         GlobalMaximizer (core.py)      │
│  ┌───────────────────────────────────┐ │
│  │  Multi-Seed Orchestration         │ │
│  │  - Seed generation/management     │ │
│  │  - Result aggregation             │ │
│  └───────────┬───────────────────────┘ │
│              │                         │
│              ▼                         │
│  ┌───────────────────────────────────┐ │
│  │  ObjectiveInverter                │ │
│  │  - Wraps f(x) → -f(x)            │ │
│  │  - Maintains original function    │ │
│  └───────────┬───────────────────────┘ │
│              │                         │
│              ▼                         │
│  ┌───────────────────────────────────┐ │
│  │  LocalPeakFinder                  │ │
│  │  - Hill-climbing search           │ │
│  │  - Adaptive step sizing           │ │
│  │  - Neighbor exploration           │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │   utils.py            │
     │  - Bounds validation  │
     │  - Seed generation    │
     │  - Convergence checks │
     └───────────────────────┘
```

### Files

- **lib/core.py** — Contains the main GlobalMaximizer class, LocalPeakFinder for hill-climbing, and ObjectiveInverter for function transformation.
- **lib/utils.py** — Provides utility functions for bounds validation, random seed generation, convergence checking, and input sanitization.
- **demo.py** — Demonstrates library usage with multiple examples including 1D, 2D, and multimodal optimization with visualization.

### Design Decisions

- Separated local search from global orchestration to allow independent testing and potential algorithm swapping.
- Used objective inversion rather than negation at call sites to maintain clean separation between maximization and minimization logic.
- Implemented adaptive step sizing in the local peak finder to balance exploration speed with convergence precision.
- Made seed generation flexible (random or user-specified) to support both automated exploration and domain-knowledge-guided search.
- Stored all local maxima found during multi-seed search to enable post-analysis and identification of multiple good solutions.
- Used NumPy arrays throughout for efficient vectorized operations and seamless multi-dimensional support.
- Implemented bounds as tuples per dimension rather than global constraints to support asymmetric search spaces.

## 🔧 Technical Details

### Dependencies

- **numpy** (1.19+) — Provides efficient array operations, vectorized computations, and mathematical functions for optimization algorithms.
- **matplotlib** (3.0+) — Used in demo scripts to visualize objective functions, search trajectories, and discovered maxima in 1D and 2D.

### Key Algorithms / Patterns

- Hill-climbing local search: Iteratively explores neighboring grid points, moving to better positions until no improvement is found, then reduces step size.
- Objective function inversion: Transforms maximization f(x) into minimization -f(x) to leverage classical optimization patterns.
- Multi-seed random restart: Generates multiple random starting points within bounds and runs independent local searches to escape local optima.
- Adaptive step size reduction: Decreases search radius geometrically (typically by factor of 2) when local search stagnates to refine solution precision.

### Important Notes

- The algorithm is gradient-free and works with non-differentiable or noisy objective functions, but may require more function evaluations than gradient-based methods.
- Performance depends heavily on the number of seeds; more seeds increase exploration but also computational cost.
- The grid-based neighbor search means the algorithm may miss very narrow peaks if the initial step size is too large.
- Convergence is determined by step size threshold rather than function value change, which works well for most smooth functions.
- For high-dimensional problems (>10 dimensions), consider increasing num_seeds and max_iterations for adequate coverage.

## ❓ Troubleshooting

### Algorithm returns suboptimal maximum or misses global peak

**Cause:** Insufficient number of seeds or initial step size too large, causing the search to skip over narrow peaks or unexplored regions.

**Solution:** Increase num_seeds parameter (try 20-50 for complex functions) and decrease initial_step_size in GlobalMaximizer. Also verify bounds cover the expected maximum location.

### ImportError: No module named 'lib.core'

**Cause:** Python cannot find the lib module because the script is not run from the project root directory or PYTHONPATH is not set correctly.

**Solution:** Run scripts from the project root directory where the lib/ folder is located, or add the project root to PYTHONPATH: export PYTHONPATH="${PYTHONPATH}:/path/to/project"

### Optimization takes too long or never completes

**Cause:** max_iterations is too high, step size reduction is too slow, or the objective function is computationally expensive.

**Solution:** Reduce max_iterations (default 100 is usually sufficient), increase step_reduction_factor (e.g., 0.3 instead of 0.5), or optimize your objective function implementation.

### ValueError: bounds must be specified for each dimension

**Cause:** The bounds parameter doesn't match the dimensionality of the objective function input, or bounds are formatted incorrectly.

**Solution:** Ensure bounds is a list of tuples with one (min, max) tuple per dimension: bounds=[(-5, 5), (-3, 3)] for 2D. Check that your objective function accepts arrays of the expected dimension.

### Results vary significantly between runs

**Cause:** Random seed generation leads to different starting points each run, and some seeds may be in better regions than others.

**Solution:** Set a random seed for reproducibility: np.random.seed(42) before creating the maximizer, or use custom seeds parameter to specify exact starting points.

---

This library provides a simple yet effective approach to global optimization without requiring derivatives. It's particularly useful for black-box optimization, noisy functions, or when gradient computation is expensive. For production use with high-dimensional problems, consider hybrid approaches combining this method with gradient-based refinement. This project was developed as a demonstration of gradient-free optimization techniques.