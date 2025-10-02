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
    batch_size::Int = BATCH_SIZE
    
    # CUDA parameters
    use_cuda::Bool = true
    threads_1d::Tuple{Int} = default_num_threads1D
    threads_2d::Tuple{Int,Int} = default_num_threads2D
    threads_3d::Tuple{Int,Int,Int} = default_num_threads3D
    
    # Validation
    function HyperSketchConfig(delta, epsilon, min_count, batch_size, use_cuda, threads_1d, threads_2d, threads_3d)
        validate_probability(delta, "delta")
        validate_probability(epsilon, "epsilon") 
        validate_min_count(min_count)
        @assert batch_size > 0 "batch_size must be positive"
        new(delta, epsilon, min_count, batch_size, use_cuda, threads_1d, threads_2d, threads_3d)
    end
end

"""
Get default configuration with optional overrides.
"""
function default_config(;kwargs...)
    HyperSketchConfig(;kwargs...)
end