#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using EpicHyperSketch
using CUDA
using DataFrames

println("=" ^ 70)
println(" " ^ 15, "EpicHyperSketch Memory Management Demo")
println("=" ^ 70)
println()

# Create a test dataset with varying complexity
activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()
filter_len = 8

println("Creating test dataset...")
for i in 1:750
    # Vary the number of features per sequence
    num_features = rand(3:12)
    features = []
    pos = 10
    for j in 1:num_features
        push!(features, (filter=rand(1:100), contribution=rand(Float32), position=pos))
        pos += filter_len + rand(1:10)
    end
    activation_dict[i] = features
end

max_active_len = EpicHyperSketch.get_max_active_len(activation_dict)
println("✓ Created $(length(activation_dict)) sequences")
println("  Max active length: $max_active_len")
println()

# Demo 1: Auto-configure with verbose output
println("=" ^ 70)
println("DEMO 1: Automatic Batch Size Configuration")
println("=" ^ 70)
result = EpicHyperSketch.auto_configure_batch_size(
    activation_dict,
    3,  # motif_size
    :Convolution;
    use_cuda=CUDA.functional(),
    verbose=true
)

# Demo 2: Compare different memory budgets
println()
println("=" ^ 70)
println("DEMO 2: Effect of Memory Budget on Batch Size")
println("=" ^ 70)
for mem_gb in [0.5, 1.0, 2.0, 4.0, 8.0]
    try
        bs = EpicHyperSketch.calculate_optimal_batch_size(
            length(activation_dict),
            max_active_len,
            3,
            :Convolution;
            target_memory_gb=mem_gb,
            use_cuda=false
        )
        num_batches = cld(length(activation_dict), bs)
        println("  Memory budget: $(mem_gb) GB → batch_size: $bs ($(num_batches) batches)")
    catch e
        println("  Memory budget: $(mem_gb) GB → ERROR: insufficient memory")
    end
end

# Demo 3: Effect of motif size on memory
println()
println("=" ^ 70)
println("DEMO 3: Effect of Motif Size on Memory Requirements")
println("=" ^ 70)
for motif_size in 2:5
    if motif_size <= max_active_len
        result = EpicHyperSketch.auto_configure_batch_size(
            activation_dict,
            motif_size,
            :Convolution;
            use_cuda=false,
            verbose=false
        )
        num_combs = binomial(max_active_len, motif_size)
        println("  Motif size $motif_size:")
        println("    Combinations: $num_combs")
        println("    Fixed memory: $(round(result.fixed_memory_mb, digits=2)) MB")
        println("    Batch size: $(result.batch_size)")
        println("    Peak memory: $(round(result.estimated_peak_memory_gb, digits=3)) GB")
        println()
    end
end

# Demo 4: Full memory report
println("=" ^ 70)
println("DEMO 4: Detailed Memory Report")
println("=" ^ 70)
config = EpicHyperSketch.default_config(
    min_count=5,
    batch_size=500,
    use_cuda=CUDA.functional()
)

EpicHyperSketch.print_memory_report(
    activation_dict,
    3,
    :Convolution,
    config
)

# Demo 5: Using auto batch size in practice
println("=" ^ 70)
println("DEMO 5: Running with Automatic Batch Size")
println("=" ^ 70)
println("Running obtain_enriched_configurations with batch_size=:auto...")
println()

config_auto = EpicHyperSketch.default_config(
    min_count=5,
    batch_size=:auto,
    use_cuda=false  # Use CPU for consistency
)

result = EpicHyperSketch.obtain_enriched_configurations_cpu(
    activation_dict;
    motif_size=3,
    filter_len=filter_len,
    config=config_auto
)

println()
println("✓ Found $(nrow(result)) enriched configurations")
println()

println("=" ^ 70)
println("Demo completed successfully!")
println("=" ^ 70)
