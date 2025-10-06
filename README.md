# EpicHyperSketch

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/EpicHyperSketch.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/EpicHyperSketch.jl/dev/)
[![Build Status](https://github.com/kchu25/EpicHyperSketch.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/EpicHyperSketch.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/EpicHyperSketch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/EpicHyperSketch.jl)

A high-performance extracting enriched motifs from data using a GPU-accelerated Count-Min Sketch.

## Installation

```julia
using Pkg
Pkg.add("EpicHyperSketch")  # When published
# Or for development:
# Pkg.develop(path="/path/to/EpicHyperSketch")
```

## Quick Start

```julia
using EpicHyperSketch
```

## Case 1: Ordinary Features

Find motifs based on feature IDs only (no position information).

### Example Data
```julia
# Dictionary: sequence_id => [feature1, feature2, ...]
activation_dict_ordinary = Dict(
    1 => [7, 3, 8, 19],
    2 => [13, 7, 28, 19],
    3 => [7, 15, 19],
    4 => [13, 9, 28, 7, 19],
    5 => [7, 12, 19],
    6 => [3, 7, 19, 13, 28],
    7 => [7, 19, 13, 28],
    8 => [13, 7, 19, 28],
    9 => [13, 28],
    10 => [13, 7, 19, 28]
)
```

### Find Enriched Motifs
```julia
# Find all 2-feature motifs that appear at least 2 times
motifs = obtain_enriched_configurations(
    activation_dict_ordinary;
    motif_size = 2,           # Look for pairs
    min_count = 7            # Must appear ≥2 times
)

println(motifs)
# Output: Set([(7, 19), (13, 28)])
# These pairs appear in multiple sequences
```

## Case 2: Convolution (Position-Aware)

Find motifs considering both filter IDs and their spatial relationships.

### Example Data
```julia
# Dictionary: sequence_id => [(filter=id, position=pos), ...]
activation_dict_conv = Dict(
    1 => [(filter=2, position=1), (filter=4, position=7)],
    2 => [(filter=2, position=3), (filter=4, position=9)], 
    3 => [(filter=1, position=19), (filter=9, position=18)],
    4 => [(filter=2, position=5), (filter=4, position=11)],
    5 => [(filter=2, position=2), (filter=4, position=8)]
)
```

### Find Enriched Spatial Motifs
```julia
# Find spatial patterns with filter_len=3
motifs = obtain_enriched_configurations(
    activation_dict_conv;
    motif_size = 2,           # Look for filter pairs  
    filter_len = 3,           # Filter length for gap calculation
    min_count = 2            # Must appear ≥2 times
)

println(motifs)
# Output: Set([(2, 3, 4)])
# Means: Filter 2, gap of 3 positions, then Filter 4
# Gap = position2 - position1 - filter_len = 7 - 1 - 3 = 3
```

## Overview

EpicHyperSketch efficiently identifies frequently occurring patterns (motifs) in large datasets by:
- Using probabilistic Count-Min Sketch for memory-efficient counting
- Supporting both ordinary features and position-aware convolution patterns
- Leveraging CUDA GPU acceleration for high performance
- Automatically falling back to CPU when GPU is unavailable


## Advanced Configuration

### Custom Configuration
```julia
using CUDA

config = HyperSketchConfig(
    min_count = 5,
    use_cuda = CUDA.functional(),     # Auto-detect GPU
    batch_size = 1000,               # Larger batches for big datasets
    threads_2d = (16, 16),           # GPU thread configuration
    delta = 0.001,                   # Count-Min Sketch error probability
    epsilon = 0.0005                 # Count-Min Sketch tolerance
)

motifs = obtain_enriched_configurations(
    activation_dict_conv;
    motif_size = 3,
    filter_len = 2, 
    config = config
)
```

### CPU-Only Mode
```julia
# Force CPU execution (useful for CI/testing)
config_cpu = HyperSketchConfig(use_cuda = false)

motifs = obtain_enriched_configurations(
    activation_dict;
    motif_size = 2,
    min_count = 3,
    config = config_cpu
)
```

### Reproducible Results
```julia
# Set a seed for reproducible hash coefficients in Count-Min Sketch
config_reproducible = HyperSketchConfig(seed = 42)

motifs1 = obtain_enriched_configurations(
    activation_dict;
    motif_size = 2,
    min_count = 2,
    config = config_reproducible
)

# Running again with same seed produces identical results
motifs2 = obtain_enriched_configurations(
    activation_dict;
    motif_size = 2,
    min_count = 2,
    config = config_reproducible
)

@assert motifs1 == motifs2  # Same results with same seed
```

## Understanding Results

### Ordinary Motifs
Results are tuples of feature IDs:
- `(7, 19)` → Feature 7 and 19

### Convolution Motifs  
Results encode spatial relationships:
- `(2, 3, 4)` → Filter 2, gap of 3, Filter 4
- `(1, 0, 5, 2, 8)` → Filter 1, gap 0, Filter 5, gap 2, Filter 8

The gap calculation: `gap = position_next - position_current - filter_len`

## Performance Notes

- **GPU Acceleration**: Automatically uses CUDA when available
- **Memory Efficient**: Count-Min Sketch reduces memory usage for large datasets  
- **Batch Processing**: Handles large datasets by processing in batches
- **Automatic Fallback**: Falls back to CPU if GPU memory insufficient

## API Reference

### Main Function
```julia
obtain_enriched_configurations(activation_dict; 
    motif_size=3, 
    filter_len=nothing, 
    min_count=1, 
    config=default_config())
```

**Parameters:**
- `activation_dict`: Input data (see examples above)
- `motif_size`: Number of features/filters in each motif
- `filter_len`: Filter length for convolution case (required for position-aware)
- `min_count`: Minimum frequency threshold  
- `config`: Configuration object for advanced settings

**Returns:** `Set{Tuple}` of enriched motifs

### Configuration
```julia
HyperSketchConfig(;
    min_count=1,
    use_cuda=true, 
    batch_size=500,
    threads_2d=(32,32),
    delta=0.0001,
    epsilon=0.00005,
    seed=nothing  # Set to an integer for reproducible results
)
```
## Speed comparison
Coming soon.

## Requirements

GPU support requires NVIDIA GPU with CUDA capability.

