# Memory Management and Automatic Batch Size Selection

## Overview

EpicHyperSketch now includes intelligent memory management that automatically calculates optimal batch sizes based on:
- Available GPU/CPU memory
- Dataset characteristics (total sequences, max active length)
- Motif size and feature type (Ordinary vs Convolution)
- Count-Min Sketch parameters (delta, epsilon)

## Why This Matters

The memory consumption of EpicHyperSketch depends on several factors:

1. **Fixed structures** (shared across all batches):
   - Combinations matrix: `C(max_active_len, motif_size)` combinations
   - Count-Min Sketch: `rows × cols` matrix (depends on delta, epsilon)
   - Hash coefficients

2. **Per-batch structures** (scales with batch_size):
   - vecRefArray: stores feature/filter data for each sequence
   - vecRefArrayContrib: stores contribution values
   - selectedCombs: stores which combinations exceed min_count

**Key insight**: As `max_active_len` increases, the combinations matrix grows exponentially (C(n,k)), which means less memory is available for batches. The optimal `batch_size` should decrease to compensate.

## Basic Usage

### Option 1: Automatic Batch Size (Recommended)

Simply use `:auto` for batch_size:

```julia
using EpicHyperSketch

# Your activation dictionary
activation_dict = Dict(...)

# Use automatic batch size selection
config = default_config(
    min_count=5,
    batch_size=:auto,  # ← Let the system choose
    use_cuda=true
)

motifs = obtain_enriched_configurations(
    activation_dict;
    motif_size=3,
    filter_len=8,
    config=config
)
```

### Option 2: Get Batch Size Recommendation

Get a recommendation with detailed memory analysis:

```julia
result = auto_configure_batch_size(
    activation_dict,
    3,  # motif_size
    :Convolution;
    use_cuda=true,
    verbose=true  # Print detailed report
)

println("Recommended batch_size: ", result.batch_size)
println("Number of batches: ", result.num_batches)
println("Peak memory estimate: ", result.estimated_peak_memory_gb, " GB")
```

Output:
```
============================================================
Auto-configured batch size
============================================================
Dataset characteristics:
  Total data points: 750
  Max active length: 8
  Motif size: 3
  Case: Convolution

Memory analysis:
  Fixed memory (sketch, combs, etc.): 2.07 MB
  Memory per data point: 0.17 KB
  Memory per batch: 2.16 MB

Recommended configuration:
  Batch size: 750
  Number of batches: 1
  Estimated peak memory: 0.002 GB
============================================================
```

### Option 3: Manual Calculation with Memory Budget

Calculate batch size for a specific memory budget:

```julia
batch_size = calculate_optimal_batch_size(
    1000,  # total sequences
    50,    # max_active_len
    3,     # motif_size
    :Convolution;
    target_memory_gb=4.0,  # Use 4GB
    use_cuda=true
)

println("Optimal batch_size for 4GB: ", batch_size)
```

## Advanced Usage

### Memory Report for Existing Configuration

Get a detailed breakdown of memory usage:

```julia
config = default_config(min_count=5, batch_size=500)

print_memory_report(
    activation_dict,
    3,  # motif_size
    :Convolution,
    config
)
```

Output:
```
======================================================================
                    MEMORY USAGE REPORT
======================================================================

Configuration:
  Dataset size: 750 sequences
  Max active length: 12
  Motif size: 3
  Case: Convolution
  Batch size: 500
  Number of batches: 2
  Backend: GPU (CUDA)

Memory breakdown:
  Fixed structures:
    - Combinations matrix: 1.03 MB
    - Count-Min Sketch (10×54366): 2.07 MB
    - Hash coefficients: 0.23 KB
    - Total fixed: 3.11 MB

  Per-batch structures (batch_size=500):
    - vecRefArray: 0.07 MB
    - vecRefArrayContrib: 0.02 MB
    - selectedCombs: 1.03 MB
    - Total per batch: 1.12 MB

Total estimated peak memory: 0.004 GB

GPU Memory:
  Available: 22.43 GB
  Total: 23.99 GB
  Estimated usage: 0.0% of available
======================================================================
```

### Custom Memory Budget and Safety Factor

```julia
batch_size = calculate_optimal_batch_size(
    5000,    # large dataset
    100,     # high max_active_len
    4,       # larger motifs
    :OrdinaryFeatures;
    target_memory_gb=8.0,    # 8GB budget
    safety_factor=0.7,       # Use only 70% (more conservative)
    min_batch_size=50,       # Don't go below 50
    max_batch_size=1000      # Cap at 1000
)
```

## How It Works

The system estimates memory usage using these formulas:

### Per Data Point Memory
```
For Convolution:
  per_point = max_active_len × 3 × 4 bytes  (vecRefArray)
            + max_active_len × 4 bytes       (vecRefArrayContrib)  
            + C(max_active_len, motif_size) × 1 byte  (selectedCombs)

For OrdinaryFeatures:
  per_point = max_active_len × 2 × 4 bytes  (vecRefArray)
            + max_active_len × 4 bytes       (vecRefArrayContrib)
            + C(max_active_len, motif_size) × 1 byte  (selectedCombs)
```

### Fixed Memory
```
Fixed = motif_size × C(max_active_len, motif_size) × 4 bytes  (combs)
      + rows × cols × 4 bytes  (sketch)
      + rows × hash_cols × 4 bytes  (hash_coeffs)

where:
  rows = ceil(log(1/delta))
  cols = ceil(e/epsilon) / rows
  hash_cols = motif_size (Ordinary) or 2×motif_size-1 (Convolution)
```

### Batch Size Calculation
```
available_memory = target_memory - fixed_memory
batch_size = floor(available_memory / per_point_memory)
```

## Best Practices

1. **Use `:auto` for most cases**: The automatic selection works well for typical workloads.

2. **Check memory reports**: Use `print_memory_report()` to understand your memory footprint.

3. **Large `max_active_len`**: When dealing with high max_active_len (>50), consider:
   - Using automatic batch sizing
   - Reducing motif_size if possible
   - Increasing available memory budget

4. **GPU Memory**: The system auto-detects GPU memory and uses 80% by default. You can adjust with `safety_factor`.

5. **Small datasets**: For small datasets where all data fits in one batch, the overhead is minimal.

## Examples by Scale

### Small Dataset (< 100 sequences)
```julia
# Fixed batch_size is fine
config = default_config(batch_size=100, min_count=5)
```

### Medium Dataset (100-10,000 sequences)
```julia
# Auto batch_size recommended
config = default_config(batch_size=:auto, min_count=5)
```

### Large Dataset (> 10,000 sequences)
```julia
# Auto with explicit memory budget
result = auto_configure_batch_size(
    activation_dict, 3, :Convolution;
    target_memory_gb=16.0,  # Specify available memory
    verbose=true
)
config = default_config(batch_size=result.batch_size, min_count=5)
```

### High `max_active_len` (> 50)
```julia
# Careful: combinations grow as C(n, k)
result = auto_configure_batch_size(
    activation_dict, 3, :Convolution;
    safety_factor=0.7,  # Be more conservative
    verbose=true
)

# Check if combinations matrix is too large
if result.fixed_memory_mb > 1000  # > 1GB fixed structures
    @warn "Large combinations matrix, consider reducing max_active_len or motif_size"
end
```

## Troubleshooting

### "Insufficient memory" error
```julia
# Reduce memory requirements:
# 1. Reduce batch_size (if set manually)
# 2. Increase target_memory_gb
# 3. Reduce motif_size (reduces combinations)
# 4. Filter activation_dict to reduce max_active_len
```

### Very small batch size
If automatic sizing gives batch_size < 100, it may indicate:
- Very high `max_active_len` leading to huge combinations matrix
- Insufficient memory budget
- Consider filtering data or reducing motif_size

### Out of GPU memory
```julia
# Force CPU mode if GPU memory insufficient
config = default_config(batch_size=:auto, use_cuda=false)
```

## API Reference

### `calculate_optimal_batch_size`
```julia
calculate_optimal_batch_size(
    total_data_points::Integer,
    max_active_len::Integer,
    motif_size::Integer,
    case::Symbol;
    target_memory_gb=nothing,      # Auto-detect if nothing
    use_cuda=true,
    safety_factor=0.8,             # Use 80% of available memory
    min_batch_size=10,
    max_batch_size=10000,
    delta=DEFAULT_CMS_DELTA,
    epsilon=DEFAULT_CMS_EPSILON
) -> Int
```

### `auto_configure_batch_size`
```julia
auto_configure_batch_size(
    activation_dict::Dict,
    motif_size::Integer,
    case::Symbol;
    verbose=false,
    kwargs...  # Passed to calculate_optimal_batch_size
) -> NamedTuple
```

### `print_memory_report`
```julia
print_memory_report(
    activation_dict::Dict,
    motif_size::Integer,
    case::Symbol,
    config::HyperSketchConfig
)
```
