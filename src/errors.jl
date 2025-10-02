# Error handling and validation utilities

"""Custom exceptions for EpicHyperSketch."""
struct HyperSketchError <: Exception
    msg::String
end

struct InvalidConfigurationError <: Exception
    msg::String
end

struct CUDAError <: Exception
    msg::String
end

"""
Macro for parameter validation with clear error messages.
"""
macro validate(condition, message)
    quote
        $(esc(condition)) || throw(InvalidConfigurationError($(esc(message))))
    end
end

"""
Check CUDA availability and requirements.
"""
function check_cuda_requirements(use_cuda::Bool)
    if use_cuda && !CUDA.functional()
        throw(CUDAError("CUDA requested but not available"))
    end
end

"""
Validate activation dictionary structure.
"""
function validate_activation_dict(dict::ActivationDict)
    isempty(dict) && throw(InvalidConfigurationError("Activation dictionary cannot be empty"))
    
    # Check for consistent value types
    first_type = typeof(first(values(dict))[1])
    for values_vec in values(dict)
        isempty(values_vec) && continue
        typeof(values_vec[1]) == first_type || 
            throw(InvalidConfigurationError("Inconsistent value types in activation dictionary"))
    end
end