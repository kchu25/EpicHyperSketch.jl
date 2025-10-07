# Large-Scale Convolution Test Documentation

## Overview
The `test_large_example_convolution.jl` file implements a large-scale test for the ConvolutionFeature case, parallel to the existing `test_large_example_ordinary.jl` test for OrdinaryFeature.

## Test Structure

### Ground Truth Motifs
The test creates 750 sequences with the following known motifs (motif_size=3, filter_len=8):

1. **Motif [7, 19, 42]** at positions [10, 20, 32]: appears 25 times
   - Distances: d12 = 20-10-8 = 2, d23 = 32-20-8 = 4

2. **Motif [13, 28, 55]** at positions [5, 18, 29]: appears 15 times
   - Distances: d12 = 18-5-8 = 5, d23 = 29-18-8 = 3

3. **Motif [3, 41, 67]** at positions [15, 25, 40]: appears 8 times
   - Distances: d12 = 25-15-8 = 2, d23 = 40-25-8 = 7

4. **Motif [22, 8, 39]** at positions [8, 20, 35]: appears 12 times
   - Distances: d12 = 20-8-8 = 4, d23 = 35-20-8 = 7

### Key Differences from Ordinary Feature Test

1. **Position Management**: ConvolutionFeature requires explicit position information, and positions must be spaced to avoid filter overlaps.
   
2. **Distance Constraints**: For two filters at positions p1 and p2 with filter_len, the distance d = p2 - p1 - filter_len must be non-negative (no overlaps).

3. **Fixed Positions**: Unlike the ordinary case where only filter combinations matter, convolution motifs must have consistent position/distance patterns to be counted as the same configuration.

4. **Column Names**: The output DataFrame uses columns `m1, m2, m3, d12, d23, start, end, data_index, contribution` (not `f1, f2, f3` as initially attempted).

## Helper Functions

### `make_convolution_features_with_positions`
Creates ConvolutionFeature entries from explicit (filter, position) tuples:
```julia
make_convolution_features_with_positions([(7, 10), (19, 20), (42, 32)], filter_len)
```

### `make_convolution_features`
Creates ConvolutionFeature entries with automatically spaced positions to avoid overlaps:
```julia
make_convolution_features([7, 19, 42], filter_len)  # Positions automatically generated
```

## Test Functions

### `test_large_example_convolution`
Main test function that:
1. Creates test dictionary with ground truth motifs
2. Verifies ground truth by counting motif occurrences
3. Runs EpicHyperSketch with various `min_count` thresholds
4. Validates that expected motifs are found

### `test_large_example_convolution_cpu`
CPU-only version for CI environments, tests fewer `min_count` values.

## Running the Tests

### Standalone execution:
```bash
# CPU only
julia --project=. test/test_large_example_convolution.jl

# GPU (if available)
EPIC_HYPERSKETCH_GPU_TESTS=true julia --project=. test/test_large_example_convolution.jl
```

### As part of test suite:
```bash
# CPU only (default)
julia --project=. -e "using Pkg; Pkg.test()"

# With GPU tests
EPIC_HYPERSKETCH_GPU_TESTS=true julia --project=. -e "using Pkg; Pkg.test()"
```

## Expected Behavior

For `min_count = 8`:
- **Expected (Theory)**: Should find all 4 motifs with count ≥ 8 
- **Actual (Current Bug)**: Finds 3 out of 4 motifs - consistently misses [22, 8, 39] (count=12)
- **Issue**: Count-Min Sketch theoretically never underestimates, so missing a motif with count=12 when min_count=8 indicates a bug in the implementation

For `min_count = 15`:
- Correctly finds 2 out of 2 expected motifs (those with count ≥ 15)
- Both [7, 19, 42] (count=25) and [13, 28, 55] (count=15) are found as expected

**Count-Min Sketch Theoretical Guarantees:**
- CMS **never underestimates**: The minimum count across all hash rows is always ≥ true count
- Hash collisions can only cause **overestimation**, never underestimation  
- Each hash row maintains independent counters that are incremented for every occurrence
- The minimum across rows gives a count that is at least the true frequency

**Current Implementation Issue:**
- The consistent absence of motif [22, 8, 39] (true count=12) suggests a bug
- Possible causes: incorrect hash calculation, invalid combination filtering, or motif generation error
- This behavior violates CMS theoretical guarantees and needs investigation
