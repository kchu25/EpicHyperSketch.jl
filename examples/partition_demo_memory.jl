# Demonstration of memory-efficient partitioned processing
# This example shows how partitioning minimizes peak memory usage

using EpicHyperSketch
using CUDA
using Random

println("="^70)
println("Memory-Efficient Partitioned Processing Demo")
println("="^70)

# Create test data with highly variable sequence lengths
Random.seed!(42)
activation_dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int32, Float32}}}}()

println("\nGenerating test data with variable sequence lengths...")

# Create 3 groups of sequences with very different lengths:
# Group 1: Short sequences (5-10 features)
for i in 1:100
    len = 5 + rand(1:5)
    activation_dict[i] = [(feature=Int32(rand(1:20)), contribution=Float32(rand())) for _ in 1:len]
end

# Group 2: Medium sequences (25-35 features)
for i in 101:200
    len = 25 + rand(1:10)
    activation_dict[i] = [(feature=Int32(rand(1:20)), contribution=Float32(rand())) for _ in 1:len]
end

# Group 3: Long sequences (50-60 features)
for i in 201:300
    len = 50 + rand(1:10)
    activation_dict[i] = [(feature=Int32(rand(1:20)), contribution=Float32(rand())) for _ in 1:len]
end

println("Created $(length(activation_dict)) sequences:")
println("  - 100 short sequences (5-10 features)")
println("  - 100 medium sequences (25-35 features)")
println("  - 100 long sequences (50-60 features)")

# Show memory efficiency of partitioning
println("\n" * "-"^70)
println("APPROACH 1: Non-Partitioned (Creates one large Record)")
println("-"^70)

# This would create one Record with all data, which could be memory-intensive
# for the long sequences
println("\nIf we processed all at once:")
println("  - Would need to accommodate max length (60) for all 300 sequences")
println("  - Batch processing constrained by longest sequences")
println("  - Peak memory usage: HIGH")

# Show what auto batch size would suggest
result1 = auto_configure_batch_size(activation_dict, 3, :OrdinaryFeatures; use_cuda=true, verbose=true)

println("\n" * "-"^70)
println("APPROACH 2: Partitioned (Sequential processing)")
println("-"^70)

# Create partitioned record - this stores partitions, not full Records
pr = create_partitioned_record(
    activation_dict, 3;
    partition_width=15,  # Group by length ranges of 15
    batch_size=:auto,
    use_cuda=true,
    seed=42,
    auto_batch_verbose=true
)

print_partition_stats(pr)

println("\n" * "-"^70)
println("Memory Efficiency Comparison")
println("-"^70)

println("\nKey differences:")
println("  1. PARTITIONED approach creates Records ONE AT A TIME")
println("     - Each partition processed separately")
println("     - Previous partition's memory freed before next")
println("     - Peak memory = max(individual partition memory)")
println()
println("  2. NON-PARTITIONED approach creates ONE LARGE Record")
println("     - All data processed together")
println("     - Batch size limited by longest sequences")
println("     - Peak memory = sum(all data)")
println()
println("  3. Partitioning allows each group to use optimal batch size:")
println("     - Short sequences: larger batches")
println("     - Long sequences: smaller batches")
println("     - Overall: better GPU utilization")

println("\n" * "-"^70)
println("Running Partitioned Pipeline")
println("-"^70)

# IMPORTANT: Use min_count=1 with partitioned processing
# Then filter the DataFrame afterwards for correct counting across partitions
println("\nNote: Using min_count=1 with partitioned processing")
println("Will filter by count afterwards to handle cross-partition motifs correctly")

# Use min_count=1 to extract all motifs
motifs = obtain_enriched_configurations_partitioned(
    activation_dict,
    motif_size=3,
    partition_width=15,
    batch_size=:auto,
    min_count=1,  # Extract all, filter later
    seed=42
)

println("\n✓ Extracted $(nrow(motifs)) total motifs")

# Now filter by desired count threshold
using DataFrames
min_count_threshold = 5
if hasproperty(motifs, :count)
    filtered_motifs = filter(row -> row.count >= min_count_threshold, motifs)
    println("✓ After filtering (count >= $min_count_threshold): $(nrow(filtered_motifs)) motifs")
else
    println("⚠ No count column found (may be empty result)")
    filtered_motifs = motifs
end

println("\n" * "="^70)
println("Key Takeaways:")
println("="^70)
println("1. Memory efficiency:")
println("   - Only partition dictionaries stored (lightweight)")
println("   - Records created on-demand during processing")
println("   - Records garbage collected after use")
println("   - Peak memory = max(single partition) + shared sketch")
println()
println("2. Counting with min_count:")
println("   - Use min_count=1 during partitioned processing")
println("   - Filter DataFrame afterwards: filter(row -> row.count >= N, motifs)")
println("   - This ensures motifs spanning multiple partitions are counted correctly")
println()
println("3. All partitions share a single CountMinSketch")
println("   - Counts accumulate across all partitions")
println("   - Memory efficient and statistically sound")
println("\nThis approach enables processing of much larger datasets")
println("within available GPU memory!")
println("="^70)
