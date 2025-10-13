# Summary: Partitioned Processing and min_count Handling

## Issue Identified

You correctly identified that the partitioned processing approach has an important consideration regarding `min_count`:

### The Problem

When using `min_count > 1` with partitioned processing:

1. **Counting phase**: All partitions count into the **shared sketch** ✓ (counts accumulate correctly)
2. **Selection phase**: Each partition's `make_selection!` checks if combinations in **that specific partition** meet `min_count` in the shared sketch
3. **Issue**: A motif appearing across multiple partitions might:
   - Have `total_count >= min_count` across all partitions
   - But `count_in_partition < min_count` in each individual partition
   - Result: The motif is **not selected** even though its total count exceeds the threshold

### Example Scenario

```
Motif [1, 5, 10] appears in the data:
- Partition 1 (length 10-20): 3 occurrences
- Partition 2 (length 21-30): 4 occurrences  
- Partition 3 (length 31-40): 2 occurrences
- Total across all partitions: 9 occurrences

With min_count=5:
- Shared sketch correctly has count=9 for this motif
- But during selection:
  - Partition 1: Only sees combinations in partition 1, count=3 < 5, not selected
  - Partition 2: Only sees combinations in partition 2, count=4 < 5, not selected
  - Partition 3: Only sees combinations in partition 3, count=2 < 5, not selected
- Result: Motif with total count=9 is MISSED!
```

## Solution Implemented

### 1. Added Warning

When `min_count > 1` is used with `obtain_enriched_configurations_partitioned`, a clear warning is displayed:

```julia
@warn """
Using min_count=$(config.min_count) with partitioned processing.

Note: Selection happens per-partition, so a motif appearing across multiple
partitions may not be selected if it doesn't meet min_count in each individual
partition. 

Recommended approach:
1. Set min_count=1 here to extract all motifs
2. Filter the resulting DataFrame afterwards:
   motifs = obtain_enriched_configurations_partitioned(..., min_count=1)
   filtered = filter(row -> row.count >= desired_min_count, motifs)

This ensures motifs spanning multiple partitions are correctly counted.
""" maxlog=1
```

### 2. Recommended Pattern

**Use `min_count=1` with partitioned processing**, then filter the DataFrame:

```julia
# Extract all motifs
motifs = obtain_enriched_configurations_partitioned(
    activation_dict,
    motif_size=3,
    partition_width=10,
    batch_size=:auto,
    min_count=1  # Don't filter during processing
)

# Filter by count afterwards
using DataFrames
filtered_motifs = filter(row -> row.count >= 5, motifs)
```

### 3. Updated Documentation

- `docs/memory_management.md`: Added prominent note and example
- `examples/partition_demo_memory.jl`: Demonstrates recommended pattern
- `PARTITIONING_REFACTOR.md`: Documents the consideration

## Why This Approach is Better

### Correctness
- Ensures motifs spanning multiple partitions are counted correctly
- Total counts are accurate (from shared sketch)
- No motifs are missed due to cross-partition distribution

### Simplicity
- Clean separation of concerns: extraction vs. filtering
- DataFrame filtering is straightforward and well-understood
- No complex communication needed between partitions

### Flexibility
- Users can apply different thresholds without reprocessing
- Can filter on other columns too (e.g., contribution)
- Easy to inspect counts before filtering

### Performance
- Minimal overhead: DataFrame filtering is very fast
- Avoids recreating Records just for selection
- Still get full memory efficiency benefits of partitioning

## Alternative Approaches Considered

### 1. Store Selection State
```julia
# Store selectedCombs from each partition
# Then merge and re-select based on total counts
```
❌ **Rejected**: Would require keeping all selectedCombs arrays in memory, defeating the purpose of sequential processing.

### 2. Two-Phase Selection
```julia
# Phase 1: Count all partitions
# Phase 2: Create all Records again and select with total counts
```
❌ **Rejected**: Would require recreating all Records, wasting computation and memory.

### 3. Communication Between Partitions
```julia
# Track which motifs appear in which partitions
# Coordinate selection across partitions
```
❌ **Rejected**: Complex, error-prone, and still requires storing cross-partition state.

## Implementation Details

### Files Modified
- `src/partition.jl`: Added warning in `obtain_enriched_configurations_partitioned`
- `docs/memory_management.md`: Added section on min_count with examples
- `examples/partition_demo_memory.jl`: Updated to demonstrate recommended pattern
- `PARTITIONING_REFACTOR.md`: Documented the consideration

### Code Changes
```julia
# In obtain_enriched_configurations_partitioned:
if config.min_count > 1
    @warn """...""" maxlog=1
end
```

### Tests
All existing tests pass. Tests use `min_count=1` as recommended.

## User Guidance

### When to Use Partitioned Processing

**Use it when:**
- Sequences have highly variable lengths (std(lengths) > 10)
- Limited GPU memory available
- Want optimal batch sizes for different length groups
- Dataset too large for single-Record approach

**Pattern:**
```julia
motifs = obtain_enriched_configurations_partitioned(
    activation_dict,
    motif_size=3,
    partition_width=10,
    batch_size=:auto,
    min_count=1  # Always use 1
)
filtered = filter(row -> row.count >= desired_threshold, motifs)
```

### When to Use Regular Processing

**Use it when:**
- All sequences have similar lengths
- Dataset easily fits in memory
- Want simplest possible workflow
- `min_count` filtering during processing is important

**Pattern:**
```julia
motifs = obtain_enriched_configurations(
    activation_dict,
    motif_size=3,
    min_count=5,  # Can use any value
    config=config
)
```

## Summary

Your observation was spot-on! The partitioned approach **does** have implications for `min_count`:

✅ **Counts are correct** (shared sketch accumulates properly)  
✅ **Selection is per-partition** (checks local combinations)  
⚠️ **Can miss cross-partition motifs** (if using min_count > 1)

**Solution**: Use `min_count=1` with partitioned processing, filter DataFrame afterwards.

This approach is:
- **Correct**: No motifs missed
- **Simple**: Clean separation of extraction and filtering  
- **Efficient**: No memory or computation overhead
- **Flexible**: Easy to adjust thresholds post-extraction

The implementation now includes a clear warning and comprehensive documentation to guide users toward this best practice.
