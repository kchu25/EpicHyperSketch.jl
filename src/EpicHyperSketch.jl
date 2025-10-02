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
const default_min_count = 25
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

# CUDA Kernel Launch Parameters
# Default number of threads for 1D CUDA kernels.
const default_num_threads1D = (128,)

# Default number of threads for 2D CUDA kernels.
const default_num_threads2D = (24, 24)

# Default number of threads for 3D CUDA kernels.
const default_num_threads3D = (8, 8, 8)



include("sketch.jl")
include("record.jl")
include("count_kernel_update.jl")


# Helper function to dispatch kernel based on case
function _launch_count_kernel!(r::Record, batch_idx::Int)
    common_args = (r.combs, r.refArray[batch_idx], r.cms.hash_coeffs, r.cms.sketch)
    threads = default_num_threads3D
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
function _launch_selection_kernel!(r::Record, batch_idx::Int, min_count::Int)
    common_args = (r.combs, r.refArray[batch_idx], r.cms.hash_coeffs, r.cms.sketch, r.selectedCombs[batch_idx])
    threads = default_num_threads2D
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
function _launch_config_kernel!(r::Record, batch_idx::Int, where_exceeds, configs)
    threads = default_num_threads1D
    blocks = ceil(IntType, length(where_exceeds))
    
    if r.case == :OrdinaryFeatures
        @cuda threads=threads blocks=blocks obtain_configs_ordinary!(where_exceeds, r.combs, r.refArray[batch_idx], configs)
    elseif r.case == :Convolution
        @cuda threads=threads blocks=blocks obtain_configs_conv!(where_exceeds, r.combs, r.refArray[batch_idx], configs, r.filter_len)
    else
        error("Unsupported case: $(r.case)")
    end
end

function count!(r::Record)
    """Execute counting on the sketch for all batches."""
    @assert r.use_cuda "count! currently only supports use_cuda=true"
    
    for batch_idx = 1:num_batches(r)
        _launch_count_kernel!(r, batch_idx)
        CUDA.synchronize()
    end
end

function make_selection!(r::Record; min_count=default_min_count)
    """Identify combinations that meet minimum count threshold."""
    for batch_idx = 1:num_batches(r)
        _launch_selection_kernel!(r, batch_idx, min_count)
        CUDA.synchronize()
    end
end

function _obtain_enriched_configurations_(r::Record)
    """Extract configurations where combinations exceed minimum count threshold."""
    @assert r.use_cuda "obtain_enriched_configurations currently only supports use_cuda=true"
    
    enriched_configs = Vector{Set{Tuple}}(undef, num_batches(r))
    
    for batch_idx = 1:num_batches(r)
        where_exceeds = findall(r.selectedCombs[batch_idx] .== true)
        
        if isempty(where_exceeds)
            enriched_configs[batch_idx] = Set{Vector{IntType}}()
        else
            configs = CuMatrix{IntType}(undef, length(where_exceeds), config_size(r))
            _launch_config_kernel!(r, batch_idx, where_exceeds, configs)
            enriched_configs[batch_idx] = Set(map(Tuple, eachrow(Array(configs))))
        end
    end
    
    return reduce(union, enriched_configs)
end


function obtain_enriched_configurations(
    activation_dict;
    min_count::IntType=1,
    delta::Float64=DEFAULT_CMS_DELTA,
    epsilon::Float64=DEFAULT_CMS_EPSILON
)
    @assert min_count > 0 "min_count must be greater than 0"
    @assert 0 < delta < 1 "delta must be in (0, 1)"
    @assert 0 < epsilon < 1 "epsilon must be in (0, 1)"

    # make record
    r = Record(activation_dict, 3; filter_len=8)
    # do the counting
    count!(r)
    # fill in the candidates that meet the min_count threshold
    make_selection!(r; min_count=min_count)

    # get the enriched configurations
    enriched_configs = _obtain_enriched_configurations_(r)
    return enriched_configs
end


export CountMinSketch, 
       Record

end
