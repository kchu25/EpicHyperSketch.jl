#!/usr/bin/env julia

"""
Demonstration of the partitioning functionality in EpicHyperSketch.

This script shows how partitioning the activation_dict by value lengths
can improve memory efficiency and enable optimal batch sizing for each partition
while sharing a single CountMinSketch across all partitions.
"""

using EpicHyperSketch
using Random
using Printf

println("\n" * "="^70)
println("EpicHyperSketch Partitioning Demonstration")
println("="^70)

# Set random seed for reproducibility
Random.seed!(42)

println("\n📊 Creating test data with highly variable sequence lengths...")
println("   This simulates real-world scenarios where sequences have different lengths.")

# Create activation dict with varying lengths
# Simulating different groups of sequences with different characteristics
activation_dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int32, Float32}}}}()

# Group 1: Short sequences (length 5-10)
println("\n   Group 1: 50 sequences with length 5-10 (short)")
for i in 1:50
    len = 5 + rand(1:5)
    activation_dict[i] = [(feature=Int32(rand(1:20)), contribution=Float32(rand())) for _ in 1:len]
end

# Group 2: Medium sequences (length 20-30)
println("   Group 2: 50 sequences with length 20-30 (medium)")
for i in 51:100
    len = 20 + rand(1:10)
    activation_dict[i] = [(feature=Int32(rand(1:20)), contribution=Float32(rand())) for _ in 1:len]
end

# Group 3: Long sequences (length 50-60)
println("   Group 3: 50 sequences with length 50-60 (long)")
for i in 101:150
    len = 50 + rand(1:10)
    activation_dict[i] = [(feature=Int32(rand(1:20)), contribution=Float32(rand())) for _ in 1:len]
end

# Group 4: Very long sequences (length 100-120)
println("   Group 4: 30 sequences with length 100-120 (very long)")
for i in 151:180
    len = 100 + rand(1:20)
    activation_dict[i] = [(feature=Int32(rand(1:20)), contribution=Float32(rand())) for _ in 1:len]
end

total_sequences = length(activation_dict)
lengths = [length(v) for v in values(activation_dict)]
println("\n   Total sequences: $total_sequences")
println("   Length range: $(minimum(lengths)) - $(maximum(lengths))")
println("   Mean length: $(round(sum(lengths)/length(lengths), digits=1))")

println("\n" * "="^70)
println("Approach 1: Standard (Non-Partitioned) Processing")
println("="^70)

println("\nThis approach treats all sequences uniformly, using max_active_len=$(maximum(lengths))")
println("This can be memory-inefficient for sequences with varying lengths.")

println("\n⚙️  Processing with standard approach...")
motifs_standard = obtain_enriched_configurations(
    activation_dict,
    motif_size=3,
    min_count=5,
    seed=42
)

println("\n✅ Standard approach results:")
println("   Found $(nrow(motifs_standard)) enriched motifs")
println("   All sequences padded to max_active_len=$(maximum(lengths))")

println("\n" * "="^70)
println("Approach 2: Partitioned Processing")
println("="^70)

println("\nThis approach partitions sequences by length and optimizes batch size per partition.")
println("A single CountMinSketch is shared across all partitions for consistent counting.")

println("\n⚙️  Creating partitioned record with partition_width=20...")
pr = create_partitioned_record(
    activation_dict, 3;
    partition_width=20,
    batch_size=:auto,
    use_cuda=true,
    seed=42,
    auto_batch_verbose=true
)

println("\n📈 Partition Statistics:")
print_partition_stats(pr)

println("\n⚙️  Processing with partitioned approach...")
config = default_config(min_count=5, seed=42)

println("\n   Phase 1: Counting...")
count!(pr, config)

println("\n   Phase 2: Selection...")
make_selection!(pr, config)

println("\n   Phase 3: Extraction...")
motifs_partitioned = obtain_enriched_configurations(pr, config)

println("\n✅ Partitioned approach results:")
println("   Found $(nrow(motifs_partitioned)) enriched motifs")
println("   Each partition optimized for its length range")

println("\n" * "="^70)
println("Comparison Summary")
println("="^70)

println("\n📊 Results:")
println("   Standard approach:    $(nrow(motifs_standard)) motifs")
println("   Partitioned approach: $(nrow(motifs_partitioned)) motifs")

# Calculate memory efficiency (estimated)
max_len = maximum(lengths)
avg_partition_len = mean(maximum([length(v) for v in values(activation_dict) 
                                  if any(length(v) in range for range in [pr.partition_ranges[i]])])
                        for i in 1:length(pr.partition_ranges))

memory_ratio = avg_partition_len / max_len
println("\n💾 Memory Efficiency:")
println("   Standard max_active_len: $max_len")
println("   Average partition max_active_len: $(round(avg_partition_len, digits=1))")
println("   Estimated memory savings: $(round((1 - memory_ratio) * 100, digits=1))%")

println("\n🎯 Key Benefits of Partitioning:")
println("   ✓ Each partition uses optimal max_active_len (no excessive padding)")
println("   ✓ Automatic batch size selection per partition")
println("   ✓ Single shared CountMinSketch for consistent counting")
println("   ✓ Better memory locality within each partition")
println("   ✓ More efficient GPU utilization")

println("\n" * "="^70)
println("Convenience Function: obtain_enriched_configurations_partitioned")
println("="^70)

println("\nFor typical use cases, you can use the convenience function:")
println("```julia")
println("motifs = obtain_enriched_configurations_partitioned(")
println("    activation_dict,")
println("    motif_size=3,")
println("    partition_width=20,")
println("    batch_size=:auto,")
println("    min_count=5")
println(")")
println("```")

println("\n⚙️  Testing convenience function...")
motifs_convenience = obtain_enriched_configurations_partitioned(
    activation_dict,
    motif_size=3,
    partition_width=20,
    batch_size=:auto,
    min_count=5,
    seed=42
)

println("\n✅ Convenience function results:")
println("   Found $(nrow(motifs_convenience)) enriched motifs")
println("   (Should match partitioned approach: $(nrow(motifs_convenience) == nrow(motifs_partitioned) ? "✓" : "✗"))")

println("\n" * "="^70)
println("When to Use Partitioning?")
println("="^70)

println("""
Partitioning is beneficial when:

1. 📏 Sequences have widely varying lengths
   - Example: Some sequences are 10 features long, others are 100+
   
2. 💾 Memory is limited
   - Partitioning reduces memory usage by avoiding excessive padding
   
3. ⚡ You want optimal batch sizes per length range
   - Different lengths may benefit from different batch sizes
   
4. 🎯 You need consistent counting across all data
   - The shared CountMinSketch ensures all partitions contribute to the same sketch

Partitioning is NOT needed when:
- All sequences have similar lengths (within ~10-20% of each other)
- You have abundant memory
- The dataset is small
""")

println("="^70)
println("Demo completed! 🎉")
println("="^70)
