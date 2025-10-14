# Memory Management Guide

EpicHyperSketch automatically manages memory by calculating optimal batch sizes based on your data and available resources.

## Quick Start

Just use `:auto` for batch size:

```julia
using EpicHyperSketch

motifs = obtain_enriched_configurations(
    activation_dict;
    motif_size=3,
    batch_size=:auto,  # Let the system decide (default)
    min_count=5
)
```

The system figures out the best batch size for your GPU/CPU memory and dataset characteristics.

## Why It Matters

Memory usage has two components:

**Fixed structures** (same regardless of batch size):
- Combinations matrix: C(max_active_len, motif_size) - grows exponentially
- Count-Min Sketch: Depends on delta, epsilon parameters

**Per-batch structures** (scales with batch_size):
- Feature arrays for each sequence in the batch
- Contribution values
- Selection masks

When `max_active_len` is large, the combinations matrix takes more memory, leaving less room for batches. The system handles this automatically by reducing batch size as needed.

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

## Partitioned Processing for Memory Efficiency

### Problem: Variable Sequence Lengths

When your `activation_dict` contains sequences with highly variable lengths, processing all data in a single `Record` is inefficient:

- **Batch size is constrained by longest sequences**: Even if most sequences are short, the batch size must be small enough to handle the longest ones.
- **Wasted GPU memory**: Short sequences don't need as much memory, but they're processed with the same (small) batch size as long sequences.
- **High peak memory usage**: All data structures exist in memory simultaneously.

### Solution: Partition by Length

The partitioned approach:
1. **Splits** `activation_dict` into partitions based on sequence length ranges
2. **Processes each partition sequentially** with its own optimal batch size
3. **Shares a single CountMinSketch** across all partitions
4. **Creates Records on-demand**, freeing memory between partitions

### Basic Usage

```julia
using EpicHyperSketch

# Your activation dictionary (with variable-length sequences)
activation_dict = Dict(...)

# Use partitioned processing with min_count=1 (recommended)
motifs = obtain_enriched_configurations_partitioned(
    activation_dict,
    motif_size=3,
    partition_width=10,      # Group by length ranges of 10
    batch_size=:auto,        # Auto-configure per partition
    min_count=1,             # Extract all motifs, filter afterwards
    filter_len=8             # For convolution case
)

# Filter by count afterwards (recommended for partitioned processing)
using DataFrames
filtered_motifs = filter(row -> row.count >= 5, motifs)
```

**Important Note on `min_count`**: 

When using partitioned processing, it's **recommended to use `min_count=1`** and filter the resulting DataFrame afterwards. This is because:

1. Selection happens per-partition during `make_selection!`
2. A motif appearing across multiple partitions might not meet `min_count` in any single partition
3. But its total count across all partitions might exceed the threshold

**Example of the issue:**
```julia
# Motif [1, 5, 10] appears:
# - 3 times in partition 1 (length 10-20)
# - 4 times in partition 2 (length 21-30)  
# - Total: 7 times

# If min_count=5 during processing:
# - Partition 1: 3 < 5, not selected ✗
# - Partition 2: 4 < 5, not selected ✗
# - Result: Motif with 7 total occurrences is missed!

# Better approach:
motifs = obtain_enriched_configurations_partitioned(..., min_count=1)
# Now the motif is extracted with its count
filtered = filter(row -> row.count >= 5, motifs)  
# Total count (7) >= 5, correctly included ✓
```

### How It Works

```julia
# 1. Partition by length
partitions, ranges = partition_by_length(activation_dict, 10)
# ranges might be: [5:14, 15:24, 25:34, 35:44, ...]

# 2. Create PartitionedRecord (lightweight, stores partitions not Records)
pr = create_partitioned_record(
    activation_dict, 3;
    partition_width=10,
    batch_size=:auto,
    use_cuda=true
)

# 3. Process sequentially (only one Record in memory at a time!)
config = default_config(min_count=5)
count!(pr, config)              # Processes partitions one by one
make_selection!(pr, config)     # Processes partitions one by one
motifs = obtain_enriched_configurations(pr, config)  # Extracts and combines
```

### Memory Efficiency

**Key insight**: With partitioned processing, peak memory is the maximum of individual partitions, NOT the sum!

```
Non-partitioned:
  Peak memory = Fixed + (all short sequences) + (all long sequences)
  
Partitioned:
  Peak memory = Fixed + max(memory per partition)
                      ≈ Fixed + (one partition's sequences)
```

### Example: Variable-Length Dataset

```julia
# Dataset with 3 groups of very different lengths:
# - 100 sequences with 5-10 features
# - 100 sequences with 25-35 features  
# - 100 sequences with 50-60 features

# Without partitioning:
# - Batch size constrained by longest (60 features)
# - All 300 sequences processed with small batches
# - Peak memory: HIGH

result_non_partitioned = auto_configure_batch_size(
    activation_dict, 3, :OrdinaryFeatures;
    use_cuda=true, verbose=true
)
# Might give: batch_size=50 (constrained by 60-feature sequences)

# With partitioning (width=15):
# - Partition 1 (5-19): batch_size could be 200 (short sequences)
# - Partition 2 (20-34): batch_size could be 100 (medium sequences)
# - Partition 3 (35-60): batch_size could be 40 (long sequences)
# - Each processed separately, memory freed between
# - Peak memory: LOW (only largest single partition)

motifs = obtain_enriched_configurations_partitioned(
    activation_dict,
    motif_size=3,
    partition_width=15,
    batch_size=:auto
)
```

### Configuration Options

```julia
create_partitioned_record(
    activation_dict,
    motif_size;
    partition_width=10,           # Length range for each partition
    batch_size=:auto,             # Auto-configure per partition
    use_cuda=true,
    filter_len=nothing,           # For convolution case
    seed=nothing,                 # Random seed for sketch
    auto_batch_verbose=false      # Print batch size details
)
```

### When to Use Partitioned Processing

**Use partitioned processing when:**
- Sequence lengths vary significantly (e.g., std(lengths) > 10)
- You have limited GPU memory
- You want optimal batch sizes for different length groups
- Dataset is too large for single-Record approach

**Skip partitioning when:**
- All sequences have similar lengths
- Dataset easily fits in memory
- You want simplest possible workflow

### Advanced: Manual Partition Control

```julia
# Create partitions manually
partitions, ranges = partition_by_length(activation_dict, 20)

println("Created $(length(partitions)) partitions:")
for (i, (partition, range)) in enumerate(zip(partitions, ranges))
    println("  Partition $i: length $range, $(length(partition)) sequences")
end

# Create PartitionedRecord with specific settings
pr = create_partitioned_record(
    activation_dict, 3;
    partition_width=20,
    batch_size=100,  # Fixed batch size for all partitions
    use_cuda=true
)

# View partition statistics
print_partition_stats(pr)
```

### Under the Hood

The `PartitionedRecord` struct stores:
- `partitions`: Vector of activation_dict subsets (lightweight)
- `shared_cms`: Single CountMinSketch shared across all partitions
- `partition_ranges`: Length ranges for each partition
- Configuration: motif_size, case, batch_size, etc.

When `count!`, `make_selection!`, or `obtain_enriched_configurations` is called:
1. Loop through partitions sequentially
2. For each partition, call `_create_record_for_partition()`
   - Creates Record with optimal batch size
   - Uses shared CountMinSketch
3. Process the Record (count, select, or extract)
4. Record goes out of scope and is garbage collected
5. Move to next partition

This ensures **only one Record exists in memory at any time**.
