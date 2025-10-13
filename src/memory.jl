# Memory management and batch size optimization

"""
    estimate_memory_per_batch(max_active_len, motif_size, case; delta=DEFAULT_CMS_DELTA, epsilon=DEFAULT_CMS_EPSILON)

Estimate memory consumption per data point in a batch for the given configuration.
Returns memory in bytes per data point.
"""
function estimate_memory_per_batch(
    max_active_len::Integer, 
    motif_size::Integer, 
    case::Symbol;
    delta::Float64=DEFAULT_CMS_DELTA,
    epsilon::Float64=DEFAULT_CMS_EPSILON
)
    # Size of reference arrays per data point
    ref_dims = refArraysDim[case]
    ref_array_per_point = max_active_len * ref_dims * sizeof(IntType)  # vecRefArray
    ref_contrib_per_point = max_active_len * sizeof(FloatType)         # vecRefArrayContrib
    
    # Size of combinations (shared across all batches)
    num_combs = binomial(max_active_len, motif_size)
    
    # Size of selectedCombs per data point
    selected_combs_per_point = num_combs * sizeof(Bool)
    
    # Total per data point
    per_point_memory = ref_array_per_point + ref_contrib_per_point + selected_combs_per_point
    
    return per_point_memory
end

"""
    estimate_fixed_memory(max_active_len, motif_size, case; delta=DEFAULT_CMS_DELTA, epsilon=DEFAULT_CMS_EPSILON)

Estimate fixed memory consumption (shared across all batches) for the given configuration.
Returns memory in bytes.
"""
function estimate_fixed_memory(
    max_active_len::Integer,
    motif_size::Integer,
    case::Symbol;
    delta::Float64=DEFAULT_CMS_DELTA,
    epsilon::Float64=DEFAULT_CMS_EPSILON
)
    # Combinations matrix
    num_combs = binomial(max_active_len, motif_size)
    combs_memory = motif_size * num_combs * sizeof(IntType)
    
    # Count-Min Sketch
    rows = cms_rows(delta)
    num_counters = cms_num_counters(rows, epsilon)
    cols = cms_cols(num_counters, rows)
    sketch_memory = rows * cols * sizeof(IntType)
    
    # Hash coefficients
    num_hash_cols = num_hash_columns(motif_size, case)
    hash_coeffs_memory = rows * num_hash_cols * sizeof(IntType)
    
    return combs_memory + sketch_memory + hash_coeffs_memory
end

"""
    calculate_optimal_batch_size(
        total_data_points, max_active_len, motif_size, case;
        target_memory_gb=nothing, use_cuda=true, safety_factor=0.8,
        delta=DEFAULT_CMS_DELTA, epsilon=DEFAULT_CMS_EPSILON
    )

Calculate optimal batch size based on available memory and data structure requirements.

# Arguments
- `total_data_points`: Total number of sequences in the dataset
- `max_active_len`: Maximum number of active features/filters per sequence
- `motif_size`: Size of motifs to search for
- `case`: `:OrdinaryFeatures` or `:Convolution`
- `target_memory_gb`: Target memory budget in GB (default: auto-detect 80% of available)
- `use_cuda`: Whether using GPU (affects available memory calculation)
- `safety_factor`: Fraction of available memory to use (default: 0.8)
- `delta`: Count-Min Sketch error probability
- `epsilon`: Count-Min Sketch error tolerance

# Returns
- Optimal batch size as an integer

# Example
```julia
batch_size = calculate_optimal_batch_size(
    1000,  # 1000 sequences
    50,    # max 50 active features
    3,     # looking for triplets
    :Convolution;
    target_memory_gb=4.0  # use 4GB
)
```
"""
function calculate_optimal_batch_size(
    total_data_points::Integer,
    max_active_len::Integer,
    motif_size::Integer,
    case::Symbol;
    target_memory_gb::Union{Nothing, Real}=nothing,
    use_cuda::Bool=true,
    safety_factor::Real=0.8,
    delta::Float64=DEFAULT_CMS_DELTA,
    epsilon::Float64=DEFAULT_CMS_EPSILON,
    min_batch_size::Integer=10,
    max_batch_size::Integer=10000
)
    @assert 0 < safety_factor <= 1 "safety_factor must be in (0, 1]"
    @assert motif_size <= max_active_len "motif_size must be <= max_active_len"
    
    # Determine available memory
    if target_memory_gb === nothing
        if use_cuda && CUDA.functional()
            # Use GPU memory
            available_bytes = CUDA.available_memory()
            target_bytes = floor(Int, available_bytes * safety_factor)
        else
            # Use CPU memory - default to conservative 4GB if not specified
            target_bytes = floor(Int, 4 * 1024^3 * safety_factor)
        end
    else
        target_bytes = floor(Int, target_memory_gb * 1024^3)
    end
    
    # Calculate fixed memory costs
    fixed_memory = estimate_fixed_memory(max_active_len, motif_size, case; delta=delta, epsilon=epsilon)
    
    # Calculate per-point memory costs
    per_point_memory = estimate_memory_per_batch(max_active_len, motif_size, case; delta=delta, epsilon=epsilon)
    
    # Available memory for batched data
    available_for_batches = target_bytes - fixed_memory
    
    if available_for_batches <= 0
        error("Insufficient memory: fixed structures alone require $(fixed_memory / 1024^3) GB, " *
              "but only $(target_bytes / 1024^3) GB available")
    end
    
    # Calculate optimal batch size
    optimal_batch_size = floor(Int, available_for_batches / per_point_memory)
    
    # Apply constraints
    optimal_batch_size = max(min_batch_size, min(optimal_batch_size, max_batch_size, total_data_points))
    
    return optimal_batch_size
end

"""
    auto_configure_batch_size(activation_dict, motif_size, case; kwargs...)

Automatically determine optimal batch size for the given activation dictionary.
This is a convenience function that analyzes the data and calls `calculate_optimal_batch_size`.

# Arguments
- `activation_dict`: The activation dictionary
- `motif_size`: Size of motifs to search for  
- `case`: `:OrdinaryFeatures` or `:Convolution`
- `kwargs...`: Additional arguments passed to `calculate_optimal_batch_size`

# Returns
- Optimal batch size and memory usage report as a NamedTuple

# Example
```julia
result = auto_configure_batch_size(activation_dict, 3, :Convolution; use_cuda=true)
println("Recommended batch_size: ", result.batch_size)
println("Memory per batch: ", result.memory_per_batch_mb, " MB")
```
"""
function auto_configure_batch_size(
    activation_dict::Dict,
    motif_size::Integer,
    case::Symbol;
    verbose::Bool=false,
    kwargs...
)
    # Analyze the data
    total_data_points = length(activation_dict)
    max_active_len = get_max_active_len(activation_dict)
    
    # Calculate optimal batch size
    batch_size = calculate_optimal_batch_size(
        total_data_points, max_active_len, motif_size, case; kwargs...
    )
    
    # Calculate memory usage for reporting
    per_point_mem = estimate_memory_per_batch(max_active_len, motif_size, case)
    fixed_mem = estimate_fixed_memory(max_active_len, motif_size, case)
    total_mem_per_batch = fixed_mem + per_point_mem * batch_size
    num_batches = cld(total_data_points, batch_size)
    
    result = (
        batch_size = batch_size,
        num_batches = num_batches,
        total_data_points = total_data_points,
        max_active_len = max_active_len,
        memory_per_point_kb = per_point_mem / 1024,
        memory_per_batch_mb = total_mem_per_batch / 1024^2,
        fixed_memory_mb = fixed_mem / 1024^2,
        estimated_peak_memory_gb = total_mem_per_batch / 1024^3
    )
    
    if verbose
        println("=" ^ 60)
        println("Auto-configured batch size")
        println("=" ^ 60)
        println("Dataset characteristics:")
        println("  Total data points: ", result.total_data_points)
        println("  Max active length: ", result.max_active_len)
        println("  Motif size: ", motif_size)
        println("  Case: ", case)
        println()
        println("Memory analysis:")
        println("  Fixed memory (sketch, combs, etc.): ", round(result.fixed_memory_mb, digits=2), " MB")
        println("  Memory per data point: ", round(result.memory_per_point_kb, digits=2), " KB")
        println("  Memory per batch: ", round(result.memory_per_batch_mb, digits=2), " MB")
        println()
        println("Recommended configuration:")
        println("  Batch size: ", result.batch_size)
        println("  Number of batches: ", result.num_batches)
        println("  Estimated peak memory: ", round(result.estimated_peak_memory_gb, digits=3), " GB")
        println("=" ^ 60)
    end
    
    return result
end

"""
    print_memory_report(activation_dict, motif_size, case, config::HyperSketchConfig)

Print a detailed memory usage report for the given configuration.
"""
function print_memory_report(
    activation_dict::Dict,
    motif_size::Integer,
    case::Symbol,
    config::HyperSketchConfig
)
    total_data_points = length(activation_dict)
    max_active_len = get_max_active_len(activation_dict)
    batch_size = config.batch_size
    
    # Calculate memory components
    per_point_mem = estimate_memory_per_batch(max_active_len, motif_size, case; 
                                               delta=config.delta, epsilon=config.epsilon)
    fixed_mem = estimate_fixed_memory(max_active_len, motif_size, case;
                                      delta=config.delta, epsilon=config.epsilon)
    
    total_mem_per_batch = fixed_mem + per_point_mem * batch_size
    num_batches = cld(total_data_points, batch_size)
    
    println()
    println("=" ^ 70)
    println(" " ^ 20, "MEMORY USAGE REPORT")
    println("=" ^ 70)
    println()
    println("Configuration:")
    println("  Dataset size: ", total_data_points, " sequences")
    println("  Max active length: ", max_active_len)
    println("  Motif size: ", motif_size)
    println("  Case: ", case)
    println("  Batch size: ", batch_size)
    println("  Number of batches: ", num_batches)
    println("  Backend: ", config.use_cuda ? "GPU (CUDA)" : "CPU")
    println()
    println("Memory breakdown:")
    println("  Fixed structures:")
    println("    - Combinations matrix: ", round(motif_size * binomial(max_active_len, motif_size) * sizeof(IntType) / 1024^2, digits=2), " MB")
    rows = cms_rows(config.delta)
    num_counters = cms_num_counters(rows, config.epsilon)
    cols = cms_cols(num_counters, rows)
    println("    - Count-Min Sketch ($rows×$cols): ", round(rows * cols * sizeof(IntType) / 1024^2, digits=2), " MB")
    println("    - Hash coefficients: ", round(rows * num_hash_columns(motif_size, case) * sizeof(IntType) / 1024, digits=2), " KB")
    println("    - Total fixed: ", round(fixed_mem / 1024^2, digits=2), " MB")
    println()
    println("  Per-batch structures (batch_size=$batch_size):")
    println("    - vecRefArray: ", round(max_active_len * refArraysDim[case] * batch_size * sizeof(IntType) / 1024^2, digits=2), " MB")
    println("    - vecRefArrayContrib: ", round(max_active_len * batch_size * sizeof(FloatType) / 1024^2, digits=2), " MB")
    println("    - selectedCombs: ", round(binomial(max_active_len, motif_size) * batch_size * sizeof(Bool) / 1024^2, digits=2), " MB")
    println("    - Total per batch: ", round((total_mem_per_batch - fixed_mem) / 1024^2, digits=2), " MB")
    println()
    println("Total estimated peak memory: ", round(total_mem_per_batch / 1024^3, digits=3), " GB")
    println()
    
    if config.use_cuda && CUDA.functional()
        available = CUDA.available_memory()
        total = CUDA.total_memory()
        println("GPU Memory:")
        println("  Available: ", round(available / 1024^3, digits=2), " GB")
        println("  Total: ", round(total / 1024^3, digits=2), " GB")
        println("  Estimated usage: ", round(100 * total_mem_per_batch / available, digits=1), "% of available")
        if total_mem_per_batch > available
            println("  ⚠ WARNING: Estimated memory exceeds available GPU memory!")
        end
    end
    
    println("=" ^ 70)
    println()
end
