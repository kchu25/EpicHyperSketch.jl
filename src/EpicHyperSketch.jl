module EpicHyperSketch

using CUDA
using Combinatorics
#

const IntType = Int32 # RTX 3090 has dedicated INT32 execution units in each SM 
const PrimeNumber = IntType(50000101)  # A large prime number for hashing
# Default error probability (delta) for Count-Min Sketch.
const DEFAULT_CMS_DELTA = 0.0001  
# Default error tolerance (epsilon) for Count-Min Sketch.
const DEFAULT_CMS_EPSILON = 0.00005  
# Default minimum count threshold for enriched configuration.
const default_min_count = 1
# Default batch_size for refArray in vecRefArray
const BATCH_SIZE = 500

# Constants for refArray indexing (features/filters data)
const FILTER_INDEX_COLUMN = 1  # Filter ID / Feature ID
const POSITION_COLUMN = 2      # Position in sequence

# Number of columns (2nd dimension) in refArray based on case
const refArraysSecondDim = Dict(
    :OrdinaryFeatures => 1,
    :Convolution => 2
)

include("types.jl")
include("errors.jl") 
include("config.jl")
include("performance.jl")
include("sketch.jl")
include("record.jl")
include("count_kernel_update.jl")


# Helper function to dispatch kernel based on case
function _launch_count_kernel!(r::Record, batch_idx::Int, config::HyperSketchConfig)
    common_args = (r.combs, r.refArray[batch_idx], r.cms.hash_coeffs, r.cms.sketch)
    threads = config.threads_3d
    blocks = ceil.(Int, get_cuda_count_tuple3d(r, batch_idx))
    
    if r.case == :OrdinaryFeatures
        @cuda threads=threads blocks=blocks count_kernel_ordinary(common_args...)
    elseif r.case == :Convolution
        @cuda threads=threads blocks=blocks count_kernel_conv(common_args..., r.filter_len)
    else
        error("Unsupported case: $(r.case)")
    end
end

# Helper function to dispatch candidate selection kernel based on case
function _launch_selection_kernel!(r::Record, batch_idx::Int, min_count::Int, config::HyperSketchConfig)
    common_args = (r.combs, r.refArray[batch_idx], r.cms.hash_coeffs, r.cms.sketch, r.selectedCombs[batch_idx])
    threads = config.threads_2d
    blocks = ceil.(Int, get_cuda_count_tuple2d(r, batch_idx))
    
    if r.case == :OrdinaryFeatures
        @cuda threads=threads blocks=blocks count_kernel_ordinary_get_candidate(common_args..., min_count)
    elseif r.case == :Convolution
        @cuda threads=threads blocks=blocks count_kernel_conv_get_candidates(common_args..., min_count)
    else
        error("Unsupported case: $(r.case)")
    end
end

# Helper function to dispatch config extraction kernel based on case
function _launch_config_kernel!(r::Record, batch_idx::Int, where_exceeds, motifs_obtained)
    threads = config.threads_1d
    blocks = ceil(IntType, length(where_exceeds))
    
    if r.case == :OrdinaryFeatures
        @cuda threads=threads blocks=blocks obtain_motifs_ordinary!(
            where_exceeds, r.combs, r.refArray[batch_idx], motifs_obtained)
    elseif r.case == :Convolution
        @cuda threads=threads blocks=blocks obtain_motifs_conv!(
            where_exceeds, r.combs, r.refArray[batch_idx], motifs_obtained, r.filter_len)
    else
        error("Unsupported case: $(r.case)")
    end
end

function count!(r::Record, config::HyperSketchConfig)
    """Execute counting on the sketch for all batches."""
    @assert r.use_cuda "count! currently only supports use_cuda=true"
    
    for batch_idx = 1:num_batches(r)
        _launch_count_kernel!(r, batch_idx, config)
        CUDA.synchronize()
    end
end

function make_selection!(r::Record, config::HyperSketchConfig)
    """Identify combinations that meet minimum count threshold."""
    for batch_idx = 1:num_batches(r)
        _launch_selection_kernel!(r, batch_idx, config.min_count, config)
        CUDA.synchronize()
    end
end

function _obtain_enriched_configurations_(r::Record, config::HyperSketchConfig)
    """Extract configurations where combinations exceed minimum count threshold."""
    @assert config.use_cuda "obtain_enriched_configurations currently only supports use_cuda=true"
    
    enriched_motifs = Vector{Set{Tuple}}(undef, num_batches(r))
    
    for batch_idx = 1:num_batches(r)
        where_exceeds = findall(r.selectedCombs[batch_idx] .== true)
        
        if isempty(where_exceeds)
            enriched_motifs[batch_idx] = Set{Vector{IntType}}()
        else
            motifs_obtained = CuMatrix{IntType}(undef, length(where_exceeds), actual_motif_size(r))
            _launch_config_kernel!(r, batch_idx, where_exceeds, motifs_obtained)
            enriched_motifs[batch_idx] = Set(map(Tuple, eachrow(Array(motifs_obtained))))
        end
    end

    return reduce(union, enriched_motifs)
end


function obtain_enriched_configurations(
    activation_dict::ActivationDict;
    motif_size::Integer=3,
    filter_len::Union{Integer,Nothing}=8,
    min_count::Integer=1, 
    config::HyperSketchConfig=default_config(min_count=min_count)
)
    # Validation
    validate_activation_dict(activation_dict)
    validate_motif_size(motif_size)
    check_cuda_requirements(config.use_cuda)
    
    # Create record with configuration
    r = Record(activation_dict, motif_size; 
               batch_size=config.batch_size,
               use_cuda=config.use_cuda, 
               filter_len=filter_len)
    
    # Execute pipeline
    count!(r, config)
    make_selection!(r, config)

    return _obtain_enriched_configurations_(r, config)
end

# Convenience method with individual parameters (backward compatibility)
# function obtain_enriched_configurations(
#     activation_dict::ActivationDict;
#     min_count::IntType=IntType(25),
#     delta::Float64=DEFAULT_CMS_DELTA,
#     epsilon::Float64=DEFAULT_CMS_EPSILON,
#     motif_size::Integer=3,
#     filter_len::Union{Integer,Nothing}=8,
#     kwargs...
# )
#     config = HyperSketchConfig(;
#         delta=delta, 
#         epsilon=epsilon, 
#         min_count=min_count,
#         kwargs...
#     )
#     return obtain_enriched_configurations(activation_dict; 
#                                         motif_size=motif_size,
#                                         filter_len=filter_len, 
#                                         config=config)
# end


export CountMinSketch, 
       Record,
       HyperSketchConfig,
       default_config,
       obtain_enriched_configurations,
       # Errors
       HyperSketchError,
       InvalidConfigurationError,
       CUDAError

end
