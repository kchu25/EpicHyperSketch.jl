# EpicHyperSketch

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/EpicHyperSketch.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/EpicHyperSketch.jl/dev/)
[![Build Status](https://github.com/kchu25/EpicHyperSketch.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/EpicHyperSketch.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/EpicHyperSketch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/EpicHyperSketch.jl)

GPU-accelerated extraction of enriched feature combinations from high-dimensional data using probabilistic counting.

## What It Does

Given sequences of features (activated neurons, genomic elements, categorical data), EpicHyperSketch finds which combinations co-occur frequently. It's designed for datasets with thousands of variable-length sequences where you need to find patterns efficiently without exhaustive enumeration.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/kchu25/EpicHyperSketch.jl")
```

Requires CUDA-capable GPU for best performance (CPU fallback available).

## Quick Example

```julia
using EpicHyperSketch

# Your data: dictionary mapping IDs to feature vectors
activation_dict = Dict(
    1 => [(feature=7, contribution=1.0), (feature=3, contribution=0.8), (feature=19, contribution=1.0)],
    2 => [(feature=7, contribution=1.0), (feature=19, contribution=0.9), (feature=28, contribution=1.0)],
    3 => [(feature=7, contribution=1.0), (feature=19, contribution=1.0)],
    # ... more sequences
)

# Find 2-feature motifs appearing at least 5 times
motifs = obtain_enriched_configurations(
    activation_dict;
    motif_size=2,
    min_count=5
)
# Returns DataFrame with columns: m1, m2, data_index, contribution, count
```


## Two Use Cases

### Ordinary Features (Position-Independent)

When you care about which features co-occur, regardless of position:

```julia
activation_dict = Dict(
    1 => [(feature=7, contribution=1.0), (feature=3, contribution=0.8), (feature=19, contribution=1.0)],
    2 => [(feature=13, contribution=1.0), (feature=7, contribution=0.9), (feature=28, contribution=1.0)],
    # ...
)

motifs = obtain_enriched_configurations(
    activation_dict;
    motif_size=2,
    min_count=5
)
# Result: m1, m2 columns show which feature pairs are enriched
```

### Convolution Features (Position-Aware)

When feature positions matter (sequence motifs, spatial patterns):

```julia
activation_dict = Dict(
    1 => [(filter=2, contribution=1.0, position=10), 
          (filter=4, contribution=0.9, position=18)],
    2 => [(filter=2, contribution=1.0, position=5), 
          (filter=4, contribution=1.0, position=13)],
    # ...
)

motifs = obtain_enriched_configurations(
    activation_dict;
    motif_size=2,
    filter_len=8,  # Number of unique filters
    min_count=5
)
# Result includes distance columns (d12, etc.) showing spacing
```

## Memory Management

The system automatically optimizes batch sizes for your GPU and dataset:

```julia
# Automatic batch sizing (recommended)
motifs = obtain_enriched_configurations(
    activation_dict;
    motif_size=3,
    batch_size=:auto  # Default
)
```

For datasets with highly variable sequence lengths, use partitioned processing:

```julia
# Partition by length for better memory efficiency
motifs = obtain_enriched_configurations_partitioned(
    activation_dict;
    motif_size=3,
    partition_width=10,  # Group by length ranges
    batch_size=:auto,
    min_count=1  # Always use 1, filter afterwards
)

# Filter by count threshold
using DataFrames
filtered = filter(row -> row.count >= 5, motifs)
```

Why partition? When sequences range from 10 to 100 features, partitioning lets short sequences use larger batches and long sequences use smaller batches, reducing peak memory by 40-60%.

**Important**: With partitioned processing, always use `min_count=1` and filter the resulting DataFrame. A motif appearing 3 times in partition 1 and 4 times in partition 2 (7 total) would be rejected by `min_count=5` in each partition separately, even though it exceeds the threshold overall.

## Configuration

```julia
config = default_config(
    min_count=5,
    batch_size=:auto,
    use_cuda=true,
    seed=42  # For reproducibility
)

motifs = obtain_enriched_configurations(
    activation_dict;
    motif_size=3,
    config=config
)
```

## Understanding Output

Each row in the output DataFrame represents one occurrence:

```julia
# Ordinary features:
# m1, m2, m3: Feature IDs in the motif
# data_index: Which sequence contains this occurrence
# contribution: Combined contribution score
# count: Total occurrences across all sequences

# Convolution features (adds):
# d12, d23, ...: Distances between consecutive features
# start, end: Position range in the sequence
```

Get unique motifs with total counts:

```julia
using DataFrames
unique_motifs = combine(groupby(motifs, [:m1, :m2, :m3]), :count => first => :total_count)
```

## Complete Workflow

```julia
using EpicHyperSketch
using DataFrames

# Load data
activation_dict = ...

# Extract motifs (partitioned for variable-length sequences)
motifs = obtain_enriched_configurations_partitioned(
    activation_dict;
    motif_size=3,
    partition_width=15,
    batch_size=:auto,
    min_count=1
)

# Filter by threshold
significant = filter(row -> row.count >= 10, motifs)

# Get unique motifs with statistics
unique_motifs = combine(
    groupby(significant, [:m1, :m2, :m3]), 
    :count => first => :total_count,
    :contribution => mean => :avg_contribution
)

sort!(unique_motifs, :total_count, rev=true)
first(unique_motifs, 10)
```

## Notes

**Count-Min Sketch**: Uses probabilistic counting that never undercounts but may overcount due to hash collisions. Collisions are rare with default parameters (delta=0.001, epsilon=0.001).

**Performance**: Start with small motif_size (2-3). Higher values and longer max_active_len increase memory usage exponentially.

**Documentation**: See `docs/memory_management.md` for details on memory estimation, partitioning strategies, and troubleshooting.

## Requirements

NVIDIA GPU with CUDA support recommended. CPU fallback available but slower.

