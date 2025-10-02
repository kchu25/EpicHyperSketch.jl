# Performance optimization utilities

"""
Macro for timing GPU operations with proper synchronization.
"""
macro gpu_time(expr)
    quote
        CUDA.synchronize()
        start_time = time()
        result = $(esc(expr))
        CUDA.synchronize()
        elapsed = time() - start_time
        (result, elapsed)
    end
end

"""
Memory usage monitoring for CUDA operations.
"""
function memory_info()
    if CUDA.functional()
        free_mem = CUDA.available_memory()
        total_mem = CUDA.total_memory()
        used_mem = total_mem - free_mem
        return (used=used_mem, free=free_mem, total=total_mem)
    else
        return nothing
    end
end

"""
Optimize CUDA kernel launch parameters based on device capabilities.
"""
function optimize_kernel_params(n_elements::Integer, default_threads::Tuple)
    if !CUDA.functional()
        return default_threads
    end
    
    device = CUDA.device()
    max_threads = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    
    # Simple optimization - could be more sophisticated
    suggested_threads = min(max_threads, max(32, n_elements ÷ 100))
    
    if length(default_threads) == 1
        return (suggested_threads,)
    elseif length(default_threads) == 2
        dim = isqrt(suggested_threads)
        return (dim, dim)
    else # 3D
        dim = round(Int, suggested_threads^(1/3))
        return (dim, dim, dim)
    end
end