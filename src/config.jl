# Configuration and parameter management

"""
Configuration parameters for EpicHyperSketch operations.
"""
Base.@kwdef struct HyperSketchConfig
    # Count-Min Sketch parameters
    delta::Float64 = DEFAULT_CMS_DELTA
    epsilon::Float64 = DEFAULT_CMS_EPSILON
    
    # Processing parameters
    min_count::IntType = IntType(default_min_count)
    batch_size::Union{Int, Symbol} = BATCH_SIZE  # Can be Integer or :auto
    
    # CUDA Kernel Launch Parameters
    use_cuda::Bool = true
    threads_1d::Tuple{Int} = (128,)
    threads_2d::Tuple{Int,Int} = (24, 24)
    threads_3d::Tuple{Int,Int,Int} = (8, 8, 8)
    
    # Reproducibility
    seed::Union{Int, Nothing} = nothing

    # Validation
    function HyperSketchConfig(delta, epsilon, min_count, batch_size, use_cuda, threads_1d, threads_2d, threads_3d, seed)
        validate_probability(delta, "delta")
        validate_probability(epsilon, "epsilon") 
        validate_min_count(min_count)
        if batch_size isa Integer
            @assert batch_size > 0 "batch_size must be positive"
        elseif batch_size isa Symbol
            @assert batch_size == :auto "batch_size must be a positive integer or :auto"
        else
            error("batch_size must be an Integer or :auto")
        end
        new(delta, epsilon, min_count, batch_size, use_cuda, threads_1d, threads_2d, threads_3d, seed)
    end
end

"""
Get default configuration with optional overrides.
"""
function default_config(;kwargs...)
    HyperSketchConfig(;kwargs...)
end