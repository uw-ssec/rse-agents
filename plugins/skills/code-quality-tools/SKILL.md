---
name: code-quality-tools
description: Automated code quality tools for scientific Python using ruff, mypy, and pre-commit hooks
---

# Code Quality Tools for Scientific Python

Master the essential code quality tools that keep scientific Python projects maintainable, consistent, and error-free. Learn how to configure **ruff** for lightning-fast linting and formatting, **mypy** for static type checking, and **pre-commit** hooks for automated quality gates. These tools help catch bugs early, enforce consistent style across teams, and make code reviews focus on logic rather than formatting.

**Key Tools:**
- **Ruff**: Ultra-fast Python linter and formatter (replaces flake8, black, isort, and more)
- **MyPy**: Static type checker for Python
- **Pre-commit**: Git hook framework for automated checks

## Quick Reference Card

### Installation & Setup
```bash
# Using pixi (recommended for scientific projects)
pixi add --feature dev ruff mypy pre-commit

# Using pip
pip install ruff mypy pre-commit

# Initialize pre-commit
pre-commit install
```

### Essential Ruff Commands
```bash
# Check code (linting)
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Format code
ruff format .

# Check and format together
ruff check --fix . && ruff format .
```

### Essential MyPy Commands
```bash
# Type check entire project
mypy src/

# Type check with strict mode
mypy --strict src/

# Type check specific file
mypy src/mymodule/analysis.py

# Generate type coverage report
mypy --html-report mypy-report src/
```

### Essential Pre-commit Commands
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run hooks on staged files only
pre-commit run

# Update hook versions
pre-commit autoupdate

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

### Quick Decision Tree

```
Need to enforce code style and catch common errors?
  YES → Use Ruff (linting + formatting)
  NO → Skip to type checking

Want to catch type-related bugs before runtime?
  YES → Add MyPy
  NO → Ruff alone is sufficient

Need to ensure checks run automatically?
  YES → Set up pre-commit hooks
  NO → Run tools manually (not recommended for teams)

Working with legacy code without type hints?
  YES → Start with Ruff only, add MyPy gradually
  NO → Use both Ruff and MyPy from the start
```

## When to Use This Skill

Use this skill when you need to establish or improve code quality practices in scientific Python projects:

- Starting a new scientific Python project and want to establish code quality standards from day one
- Maintaining existing research code that needs consistency and error prevention
- Collaborating with multiple contributors who need automated style enforcement
- Preparing code for publication or package distribution
- Catching bugs early through static type checking before runtime
- Automating code reviews to focus on logic rather than style
- Integrating with CI/CD for automated quality checks
- Migrating from older tools like black, flake8, or isort to modern alternatives

## Core Concepts

### 1. Ruff: The All-in-One Linter and Formatter

**Ruff** is a blazingly fast Python linter and formatter written in Rust that replaces multiple tools you might be using today.

**What Ruff Replaces:**
- flake8 (linting)
- black (formatting)
- isort (import sorting)
- pyupgrade (syntax modernization)
- pydocstyle (docstring linting)
- And 50+ other tools

**Why Ruff for Scientific Python:**

Ruff is 10-100x faster than traditional tools, which matters when you have large codebases with thousands of lines of numerical code. Instead of managing multiple configuration files and tool versions, you get a single tool that handles everything. Ruff can auto-fix most issues automatically, saving time during development. It includes NumPy-aware docstring checking, understanding the conventions used throughout the scientific Python ecosystem. Best of all, it's compatible with existing black and flake8 configurations, making migration straightforward.

**Example:**
```python
# Before ruff format
import sys
import os
import numpy as np

def calculate_mean(data):
    return np.mean(data)

# After ruff format
import os
import sys

import numpy as np


def calculate_mean(data):
    return np.mean(data)
```

Ruff automatically organizes imports (standard library, third party, local) and applies consistent formatting.

### 2. MyPy: Static Type Checking

**MyPy** analyzes type hints to catch errors before your code ever runs. This is especially valuable in scientific computing where dimension mismatches and type errors can lead to subtle bugs in numerical calculations.

**Example of what MyPy catches:**

```python
import numpy as np
from numpy.typing import NDArray

def calculate_mean(data: NDArray[np.float64]) -> float:
    """Calculate mean of array."""
    return float(np.mean(data))

# MyPy catches this error at type-check time:
result: int = calculate_mean(np.array([1.0, 2.0, 3.0]))
# Error: Incompatible types (expression has type "float", variable has type "int")
```

**Benefits for Scientific Code:**

Type hints catch dimension mismatches in array operations before you run expensive computations. They validate function signatures, ensuring you pass the right types to numerical functions. Type hints serve as documentation, making it clear what types functions expect and return. They prevent None-related bugs that can crash long-running simulations. Modern IDEs use type hints to provide better autocomplete and inline documentation.

### 3. Pre-commit: Automated Quality Gates

**Pre-commit** runs checks automatically before each commit, ensuring code quality standards are maintained without manual intervention.

**How it works:**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.0
    hooks:
      - id: mypy
```

**Workflow:**
1. Developer runs `git commit`
2. Pre-commit automatically runs ruff, mypy, and other checks
3. If checks fail, commit is blocked
4. Developer fixes issues and commits again
5. Once all checks pass, commit succeeds

This ensures that code quality issues are caught immediately, before they enter the codebase.


## Decision Trees

### Choosing Between Ruff and Legacy Tools

```
Already using black + flake8 + isort?
  YES → Migrate to Ruff (single tool, much faster)
    Ruff is compatible with black formatting
  NO → Start with Ruff directly

Need custom linting rules?
  YES → Check if Ruff supports them (700+ rules available)
    Supported → Use Ruff
    Not supported → Consider pylint as supplement
  NO → Ruff covers 99% of use cases

Performance matters (large codebase)?
  Always → Ruff is 10-100x faster
```

### MyPy Strictness Levels

```
Starting a new project?
  YES → Use --strict mode from day one
  NO → Adding types to existing code?
    Start with basic mypy (no flags)
    Add --check-untyped-defs
    Add --disallow-untyped-defs
    Eventually reach --strict

Scientific library with complex NumPy types?
  YES → Install numpy type stubs: pip install types-numpy
  NO → Standard mypy is sufficient
```

## Patterns and Examples

### Pattern 1: Basic Ruff Configuration

Configure ruff in `pyproject.toml` for your scientific Python project:

```toml
[tool.ruff]
# Target Python 3.10+
target-version = "py310"

# Line length (match black default)
line-length = 88

# Exclude common directories
exclude = [
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
]

[tool.ruff.lint]
# Enable rule sets
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort (import sorting)
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "NPY",    # NumPy-specific rules
    "PD",     # pandas-vet
]

# Ignore specific rules
ignore = [
    "E501",   # Line too long (handled by formatter)
]

# Allow autofix for all enabled rules
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
# Ignore imports in __init__.py
"__init__.py" = ["F401"]
# Allow print statements in scripts
"scripts/*.py" = ["T201"]

[tool.ruff.format]
# Use double quotes
quote-style = "double"

# Indent with spaces
indent-style = "space"
```

**Usage:**
```bash
# Check and fix
ruff check --fix .

# Format
ruff format .
```

### Pattern 2: MyPy Configuration for Scientific Python

Configure mypy in `pyproject.toml` with appropriate strictness for scientific code:

```toml
[tool.mypy]
# Python version
python_version = "3.10"

# Strictness options (start lenient, increase gradually)
check_untyped_defs = true
disallow_untyped_defs = false  # Set to true when ready
warn_return_any = true
warn_unused_configs = true
warn_redundant_casts = true

# Output options
show_error_codes = true
pretty = true

# Ignore missing imports for packages without type stubs
[[tool.mypy.overrides]]
module = [
    "scipy.*",
    "matplotlib.*",
]
ignore_missing_imports = true
```

**Install type stubs for common libraries:**
```bash
pixi add --feature dev types-requests types-PyYAML
# NumPy and pandas have built-in type hints (Python 3.9+)
```

**Example typed scientific function:**
```python
import numpy as np
from typing import Optional
from numpy.typing import NDArray

def normalize_data(
    data: NDArray[np.float64],
    method: str = "zscore",
    axis: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Normalize numerical data.
    
    Parameters
    ----------
    data : NDArray[np.float64]
        Input data array.
    method : str, default "zscore"
        Normalization method: "zscore" or "minmax".
    axis : int, optional
        Axis along which to normalize.
    
    Returns
    -------
    NDArray[np.float64]
        Normalized data.
    
    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method == "zscore":
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        return (data - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown method: {method}")
```


### Pattern 3: Pre-commit Configuration

Set up pre-commit hooks to automatically enforce quality standards:

```yaml
# .pre-commit-config.yaml
# See https://pre-commit.com for more information
repos:
  # Ruff linter and formatter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      # Run the linter
      - id: ruff
        args: [--fix]
      # Run the formatter
      - id: ruff-format

  # MyPy type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-requests
          - types-PyYAML
        args: [--ignore-missing-imports]

  # General pre-commit hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-merge-conflict

  # Jupyter notebook cleaning
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
```

**Setup:**
```bash
# Install pre-commit
pixi add --feature dev pre-commit

# Install git hooks
pre-commit install

# Run on all files (first time)
pre-commit run --all-files
```

### Pattern 4: Pixi Integration

Integrate quality tools with pixi for reproducible development environments:

```toml
[project]
name = "my-science-project"
version = "0.1.0"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
]

[tool.pixi.project]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]

[tool.pixi.dependencies]
python = ">=3.10"
numpy = ">=1.24"
pandas = ">=2.0"

[tool.pixi.feature.dev.dependencies]
ruff = ">=0.6.0"
mypy = ">=1.11"
pre-commit = ">=3.5"
pytest = ">=7.0"

[tool.pixi.feature.dev.tasks]
# Linting and formatting
lint = "ruff check ."
format = "ruff format ."
check = { depends-on = ["lint", "format"] }

# Type checking
typecheck = "mypy src/"

# Run all quality checks
quality = { depends-on = ["check", "typecheck"] }

# Testing
test = "pytest tests/"

# Full validation (run before committing)
validate = { depends-on = ["quality", "test"] }
```

**Usage:**
```bash
# Run quality checks
pixi run quality

# Run full validation
pixi run validate

# Format code
pixi run format

# Type check
pixi run typecheck
```

### Pattern 5: CI/CD Integration (GitHub Actions)

Ensure quality checks run automatically in continuous integration:

```yaml
# .github/workflows/quality.yml
name: Code Quality

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          pip install ruff mypy pytest
          pip install -e .
      
      - name: Run Ruff
        run: |
          ruff check .
          ruff format --check .
      
      - name: Run MyPy
        run: mypy src/
      
      - name: Run tests
        run: pytest tests/
```

### Pattern 6: Gradual Type Hint Adoption

Add type hints incrementally to existing scientific code:

**Step 1: Start with function signatures**
```python
import numpy as np
from numpy.typing import NDArray

def calculate_statistics(data: NDArray) -> dict:
    """Calculate basic statistics."""
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
    }
```

**Step 2: Add return type details**
```python
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray

class Statistics(TypedDict):
    mean: float
    std: float
    min: float
    max: float

def calculate_statistics(data: NDArray) -> Statistics:
    """Calculate basic statistics."""
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }
```

**Step 3: Add internal variable types (optional)**
```python
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray

class Statistics(TypedDict):
    mean: float
    std: float
    min: float
    max: float

def calculate_statistics(data: NDArray) -> Statistics:
    """Calculate basic statistics."""
    mean_val: float = float(np.mean(data))
    std_val: float = float(np.std(data))
    min_val: float = float(np.min(data))
    max_val: float = float(np.max(data))
    
    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
    }
```


### Pattern 7: NumPy Array Type Hints

Use numpy.typing for proper array annotations in scientific code:

```python
import numpy as np
from numpy.typing import NDArray

# Generic array
def process_array(data: NDArray) -> NDArray:
    """Process numerical array."""
    return data * 2

# Specific dtype
def process_float_array(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Process float64 array."""
    return data * 2.0

# Multiple dimensions
Vector = NDArray[np.float64]  # 1D array
Matrix = NDArray[np.float64]  # 2D array

def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two matrices."""
    return np.matmul(a, b)

# More specific shape hints
def normalize_vector(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize a vector to unit length.
    
    Parameters
    ----------
    v : NDArray[np.float64]
        Input vector of shape (n,).
    
    Returns
    -------
    NDArray[np.float64]
        Normalized vector of shape (n,).
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
```

### Pattern 8: Handling Optional and Union Types

Properly type functions with optional parameters and multiple accepted types:

```python
import numpy as np
from typing import Optional, Union
from pathlib import Path
from numpy.typing import NDArray

def load_data(
    filepath: Union[str, Path],
    delimiter: str = ",",
    skip_rows: Optional[int] = None
) -> NDArray:
    """
    Load data from file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to data file.
    delimiter : str, default ","
        Column delimiter.
    skip_rows : int, optional
        Number of rows to skip at start.
    
    Returns
    -------
    NDArray
        Loaded data array.
    """
    # Convert to Path if string
    path = Path(filepath) if isinstance(filepath, str) else filepath
    
    # Load with optional skip_rows
    kwargs = {"delimiter": delimiter}
    if skip_rows is not None:
        kwargs["skiprows"] = skip_rows
    
    return np.loadtxt(path, **kwargs)
```

### Pattern 9: Ruff Rule Selection for Scientific Python

Configure ruff with rules appropriate for scientific computing:

```toml
[tool.ruff.lint]
select = [
    # Essential
    "E",      # pycodestyle errors
    "F",      # pyflakes
    "I",      # isort
    
    # Code quality
    "B",      # flake8-bugbear (common bugs)
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade (modern syntax)
    
    # Scientific Python specific
    "NPY",    # NumPy-specific rules
    "PD",     # pandas-vet
    
    # Documentation
    "D",      # pydocstyle (docstrings)
    
    # Type hints
    "ANN",    # flake8-annotations
]

# Customize docstring rules for NumPy style
[tool.ruff.lint.pydocstyle]
convention = "numpy"

# Common rules to ignore in scientific code
ignore = [
    "E501",   # Line too long (formatter handles this)
    "ANN101", # Missing type annotation for self
    "ANN102", # Missing type annotation for cls
    "D100",   # Missing docstring in public module (optional)
    "D104",   # Missing docstring in public package (optional)
]
```

### Pattern 10: Fixing Common Ruff Warnings

Learn to fix the most common issues ruff identifies:

**F401: Unused import**
```python
# Before
import numpy as np
import pandas as pd  # Not used

# After
import numpy as np
```

**F841: Unused variable**
```python
# Before
def process_data(data):
    result = expensive_computation(data)
    return data  # Oops, should return result

# After
def process_data(data):
    result = expensive_computation(data)
    return result
```

**E711: Comparison to None**
```python
# Before
if value == None:
    pass

# After
if value is None:
    pass
```

**B006: Mutable default argument**
```python
# Before (dangerous!)
def append_data(value, data=[]):
    data.append(value)
    return data

# After
def append_data(value, data=None):
    if data is None:
        data = []
    data.append(value)
    return data
```

**NPY002: Legacy NumPy random**
```python
# Before (old style)
import numpy as np
data = np.random.rand(100)

# After (new style, better for reproducibility)
import numpy as np
rng = np.random.default_rng(seed=42)
data = rng.random(100)
```


## Best Practices Checklist

### Initial Setup
- Install ruff, mypy, and pre-commit in dev environment
- Create `pyproject.toml` with tool configurations
- Create `.pre-commit-config.yaml`
- Run `pre-commit install` to enable git hooks
- Run `pre-commit run --all-files` to check existing code
- Add quality check tasks to pixi configuration

### Configuration
- Set appropriate Python version target
- Enable NumPy-specific rules (NPY) for scientific code
- Configure NumPy-style docstring checking
- Set up per-file ignores for special cases (__init__.py, scripts)
- Configure mypy strictness appropriate for project maturity
- Install type stubs for third-party libraries

### Workflow Integration
- Add quality checks to CI/CD pipeline
- Document quality standards in CONTRIBUTING.md
- Create pixi tasks for common quality checks
- Set up IDE integration (VS Code, PyCharm)
- Configure editor to run ruff on save
- Add quality check badge to README

### Team Practices
- Run `ruff check --fix` before committing
- Run `ruff format` before committing
- Address mypy errors (don't use `# type: ignore` without reason)
- Review pre-commit failures before using `--no-verify`
- Keep pre-commit hooks updated (`pre-commit autoupdate`)
- Add type hints to new functions
- Gradually add types to existing code

### Maintenance
- Update ruff regularly (fast-moving project)
- Update pre-commit hook versions monthly
- Review and adjust ignored rules as project matures
- Increase mypy strictness gradually
- Monitor CI/CD for quality check failures
- Refactor code flagged by quality tools

## Common Issues and Solutions

### Issue 1: Ruff and Black Formatting Conflicts

**Problem:** Using both ruff format and black causes conflicts.

**Solution:** Choose one formatter. Ruff format is compatible with black's style:
```toml
[tool.ruff.format]
# Use black-compatible formatting
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

Remove black from dependencies and pre-commit hooks.

### Issue 2: MyPy Can't Find Imports

**Problem:** `error: Cannot find implementation or library stub for module named 'scipy'`

**Solution:** Install type stubs or ignore missing imports:
```toml
[[tool.mypy.overrides]]
module = ["scipy.*", "matplotlib.*"]
ignore_missing_imports = true
```

Or install stubs:
```bash
pixi add --feature dev types-requests types-PyYAML
```

### Issue 3: Pre-commit Hooks Too Slow

**Problem:** Pre-commit takes too long on large codebases.

**Solution:** 

Use ruff instead of multiple tools (much faster). Limit hooks to staged files only (default behavior). Skip expensive checks in pre-commit, run in CI instead by removing mypy from `.pre-commit-config.yaml` and keeping it in CI workflow.

### Issue 4: Too Many Ruff Errors on Legacy Code

**Problem:** Hundreds of ruff errors on existing codebase.

**Solution:** Gradual adoption strategy:
```bash
# 1. Start with auto-fixable issues only
ruff check --fix .

# 2. Add baseline to ignore existing issues
ruff check --add-noqa .

# 3. Fix new code going forward
# 4. Gradually remove # noqa comments
```

### Issue 5: Type Hints Break at Runtime

**Problem:** Code with type hints fails with `NameError` in Python < 3.10.

**Solution:** Use `from __future__ import annotations`:
```python
from __future__ import annotations  # Must be first import

import numpy as np

def process(data: np.ndarray) -> np.ndarray:
    """This works in Python 3.7+"""
    return data * 2
```

### Issue 6: MyPy Errors in Test Files

**Problem:** MyPy complains about pytest fixtures and dynamic test generation.

**Solution:** Configure mypy to be lenient with tests:
```toml
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

### Issue 7: Ruff Conflicts with Project Style

**Problem:** Team prefers single quotes, but ruff uses double quotes.

**Solution:** Configure ruff to match team preferences:
```toml
[tool.ruff.format]
quote-style = "single"
```

### Issue 8: Pre-commit Fails in CI

**Problem:** Pre-commit hooks pass locally but fail in CI.

**Solution:** Ensure consistent environments:
```yaml
# In CI, use same Python version and dependencies
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: "3.11"  # Match local version

# Or use pre-commit's CI action
- uses: pre-commit/action@v3.0.0
```


## Integration with Other Tools

### VS Code Integration

Install extensions for seamless integration with your editor:

**Extensions:**
- Ruff (charliermarsh.ruff)
- Mypy Type Checker (ms-python.mypy-type-checker)

**Settings (`.vscode/settings.json`):**
```json
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  "ruff.lint.args": ["--config=pyproject.toml"],
  "mypy-type-checker.args": ["--config-file=pyproject.toml"]
}
```

### PyCharm Integration

**Ruff:**
1. Install Ruff plugin from marketplace
2. Configure: Settings → Tools → Ruff
3. Enable "Run ruff on save"

**MyPy:**
1. Install Mypy plugin
2. Configure: Settings → Tools → Mypy
3. Set mypy executable path

### Jupyter Notebook Integration

Use nbqa to run quality tools on notebooks:

```bash
# Install nbqa
pixi add --feature dev nbqa

# Run ruff on notebooks
nbqa ruff notebooks/

# Run mypy on notebooks
nbqa mypy notebooks/
```

**Pre-commit config for notebooks:**
```yaml
repos:
  - repo: https://github.com/nbQA-dev/nbQA
    rev: 1.8.5
    hooks:
      - id: nbqa-ruff
        args: [--fix]
      - id: nbqa-mypy
```

### pytest Integration

Type checking in tests ensures your test code is also correct:

```python
import numpy as np
from numpy.typing import NDArray

def test_normalize_data():
    """Test data normalization."""
    data: NDArray[np.float64] = np.array([1.0, 2.0, 3.0])
    result = normalize_data(data)
    
    # MyPy ensures types match
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
```

### Documentation Integration

Ruff checks docstrings for completeness and correctness:

```python
def calculate_mean(data: np.ndarray) -> float:
    """
    Calculate arithmetic mean.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array.
    
    Returns
    -------
    float
        Mean value.
    
    Examples
    --------
    >>> calculate_mean(np.array([1, 2, 3]))
    2.0
    """
    return float(np.mean(data))
```

Ruff validates docstring presence, NumPy-style formatting, parameter documentation matches signature, and return type documentation.

## Real-World Examples

### Example 1: Complete Scientific Python Project Setup

Set up a new project with all quality tools configured:

**Project structure:**
```
my-science-project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── analysis.py
│       └── visualization.py
├── tests/
│   └── test_analysis.py
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

**pyproject.toml:**
```toml
[project]
name = "my-science-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "matplotlib>=3.7",
]

[tool.pixi.project]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-64", "osx-arm64"]

[tool.pixi.dependencies]
python = "3.11.*"
numpy = ">=1.24"
pandas = ">=2.0"
matplotlib = ">=3.7"

[tool.pixi.feature.dev.dependencies]
ruff = ">=0.6.0"
mypy = ">=1.11"
pre-commit = ">=3.5"
pytest = ">=7.0"
pytest-cov = ">=4.0"

[tool.pixi.feature.dev.tasks]
lint = "ruff check ."
format = "ruff format ."
typecheck = "mypy src/"
test = "pytest tests/ --cov=src/"
quality = { depends-on = ["lint", "format", "typecheck"] }
all = { depends-on = ["quality", "test"] }

[tool.ruff]
target-version = "py310"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "C4", "NPY", "D"]
ignore = ["E501", "D100", "D104"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.mypy]
python_version = "3.10"
check_untyped_defs = true
warn_return_any = true
warn_unused_configs = true
show_error_codes = true
```

**Usage:**
```bash
# Setup
pixi install
pre-commit install

# Development workflow
pixi run format      # Format code
pixi run lint        # Check for issues
pixi run typecheck   # Type check
pixi run test        # Run tests
pixi run all         # Run everything

# Before committing (automatic via pre-commit)
git commit -m "Add new analysis function"
```

### Example 2: Adding Types to Existing Scientific Code

Transform untyped code into well-typed, documented code:

**Before (no types):**
```python
import numpy as np

def calculate_correlation(x, y, method="pearson"):
    """Calculate correlation between two arrays."""
    if method == "pearson":
        return np.corrcoef(x, y)[0, 1]
    elif method == "spearman":
        from scipy.stats import spearmanr
        return spearmanr(x, y)[0]
    else:
        raise ValueError(f"Unknown method: {method}")
```

**After (with types):**
```python
import numpy as np
from numpy.typing import NDArray
from typing import Literal

CorrelationMethod = Literal["pearson", "spearman"]

def calculate_correlation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    method: CorrelationMethod = "pearson"
) -> float:
    """
    Calculate correlation between two arrays.
    
    Parameters
    ----------
    x : NDArray[np.float64]
        First data array.
    y : NDArray[np.float64]
        Second data array.
    method : {"pearson", "spearman"}, default "pearson"
        Correlation method to use.
    
    Returns
    -------
    float
        Correlation coefficient.
    
    Raises
    ------
    ValueError
        If method is not recognized.
    
    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([2.0, 4.0, 6.0])
    >>> calculate_correlation(x, y)
    1.0
    """
    if method == "pearson":
        corr_matrix: NDArray[np.float64] = np.corrcoef(x, y)
        return float(corr_matrix[0, 1])
    elif method == "spearman":
        from scipy.stats import spearmanr
        result = spearmanr(x, y)
        return float(result.statistic)
    else:
        raise ValueError(f"Unknown method: {method}")
```

**Benefits:**

MyPy catches invalid method names at type-check time. IDE provides autocomplete for method parameter. Clear documentation of expected types. Runtime errors caught before execution.


### Example 3: Pre-commit Hook Workflow

See how pre-commit catches issues before they enter the codebase:

**Scenario: Developer commits code with issues**

```bash
$ git add src/analysis.py
$ git commit -m "Add new analysis function"

# Pre-commit runs automatically
ruff....................................................................Failed
hook id: ruff
exit code: 1

src/analysis.py:15:1: F401 [*] `numpy` imported but unused
src/analysis.py:23:5: E711 Comparison to `None` should be `cond is None`
Found 2 errors.

mypy....................................................................Failed
hook id: mypy
exit code: 1

src/analysis.py:30: error: Incompatible return value type (got "None", expected "float")

# Fix the issues
$ ruff check --fix src/analysis.py  # Auto-fix F401, E711
$ # Manually fix mypy error

# Commit again
$ git commit -m "Add new analysis function"

ruff....................................................................Passed
mypy....................................................................Passed
[feature/new-analysis abc123] Add new analysis function
 1 file changed, 25 insertions(+)
```

## Migration Guides

### Migrating from Black + Flake8 + isort

Replace multiple tools with ruff for better performance and simpler configuration:

**Step 1: Install ruff**
```bash
pixi add --feature dev ruff
```

**Step 2: Convert configuration**
```toml
# Old: pyproject.toml
[tool.black]
line-length = 88

[tool.isort]
profile = "black"

# Old: setup.cfg
[flake8]
max-line-length = 88

# New: pyproject.toml
[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I"]  # pycodestyle, pyflakes, isort
```

**Step 3: Update pre-commit**
```yaml
# Remove these
# - repo: https://github.com/psf/black
# - repo: https://github.com/pycqa/flake8
# - repo: https://github.com/pycqa/isort

# Add this
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.6.0
  hooks:
    - id: ruff
      args: [--fix]
    - id: ruff-format
```

**Step 4: Remove old tools**
```bash
pixi remove --feature dev black flake8 isort
```

### Migrating from Pylint

Ruff covers most pylint rules with better performance:

```toml
[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "C90", # mccabe complexity
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "A",   # flake8-builtins
    "C4",  # flake8-comprehensions
    "PL",  # pylint rules
]
```

Keep pylint only if you need specific rules:
```bash
# Check what pylint rules you use
pylint --list-msgs

# See if ruff supports them
ruff rule <rule-code>
```

## Resources and References

### Official Documentation
- **Ruff**: https://docs.astral.sh/ruff/
- **MyPy**: https://mypy.readthedocs.io/
- **Pre-commit**: https://pre-commit.com/
- **NumPy Typing**: https://numpy.org/devdocs/reference/typing.html

### Ruff Resources
- Rule reference: https://docs.astral.sh/ruff/rules/
- Configuration: https://docs.astral.sh/ruff/configuration/
- Editor integrations: https://docs.astral.sh/ruff/integrations/
- Migration guide: https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8

### MyPy Resources
- Type hints cheat sheet: https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
- Common issues: https://mypy.readthedocs.io/en/stable/common_issues.html
- Running mypy: https://mypy.readthedocs.io/en/stable/running_mypy.html
- Type stubs: https://mypy.readthedocs.io/en/stable/stubs.html

### Pre-commit Resources
- Supported hooks: https://pre-commit.com/hooks.html
- Creating hooks: https://pre-commit.com/index.html#creating-new-hooks
- CI integration: https://pre-commit.ci/

### Scientific Python Resources
- Scientific Python Development Guide: https://learn.scientific-python.org/development/
- NumPy documentation style: https://numpydoc.readthedocs.io/
- Type hints for scientific code: https://numpy.org/devdocs/reference/typing.html

### Community Examples
- Scientific Python Cookie: https://github.com/scientific-python/cookie
- NumPy: https://github.com/numpy/numpy (see pyproject.toml)
- SciPy: https://github.com/scipy/scipy (see pyproject.toml)
- Pandas: https://github.com/pandas-dev/pandas (see pyproject.toml)

## Quick Start Template

Copy-paste starter configuration for immediate use:

```toml
# pyproject.toml
[tool.ruff]
target-version = "py310"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "B", "NPY"]
ignore = ["E501"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.mypy]
python_version = "3.10"
check_untyped_defs = true
warn_return_any = true
show_error_codes = true
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
```

```bash
# Setup commands
pixi add --feature dev ruff mypy pre-commit
pre-commit install
pre-commit run --all-files
```

## Summary

Code quality tools are essential for maintaining scientific Python projects. Ruff provides fast, comprehensive linting and formatting. MyPy catches type errors before runtime. Pre-commit automates quality checks in your workflow.

**Key takeaways:**

Start with ruff for immediate impact as it replaces multiple tools with a single fast solution. Add mypy gradually as you add type hints to catch bugs early. Use pre-commit to enforce standards automatically without manual intervention. Integrate with pixi for reproducible development environments. Configure tools in pyproject.toml for centralized management. Run quality checks in CI/CD to maintain standards across the team.

**Next steps:**

Set up ruff and pre-commit in your project today. Add type hints to new functions you write. Gradually increase mypy strictness as your codebase matures. Share configurations with your team for consistency. Integrate quality checks into your development workflow.

Quality tools save time by catching errors early and maintaining consistency across your scientific codebase. They make code reviews more productive by automating style discussions, allowing reviewers to focus on scientific correctness and algorithmic choices rather than formatting details.
