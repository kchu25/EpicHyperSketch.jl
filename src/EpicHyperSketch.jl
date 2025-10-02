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


function count!(r::Record)
    # execute the counting on the sketch
    @assert r.use_cuda "count! currently only supports use_cuda=true"
    for i = 1:num_batches(r)
        if r.case == :OrdinaryFeatures
            @cuda threads=default_num_threads3D blocks=ceil.(
                Int, get_cuda_count_tuple3d(r, i)) count_kernel_ordinary(
                    r.combs, 
                    r.refArray[i], 
                    r.cms.hash_coeffs, 
                    r.cms.sketch);
        elseif r.case == :Convolution
            @cuda threads=default_num_threads3D blocks=ceil.(
                    Int, get_cuda_count_tuple3d(r, i)) count_kernel_conv(
                        r.combs, 
                        r.refArray[i], 
                        r.cms.hash_coeffs, 
                        r.cms.sketch, 
                        r.filter_len);
        else
            error("Unsupported case: $(r.case)")
        end
        CUDA.synchronize()
    end
end


function make_selection!(r::Record;
        min_count=default_min_count)
    # get the placeholder_count
    for i = 1:num_batches(r)
        # @info "placeholder_count before: $(sum(r.placeholder_count[i]))"
        if r.case == :OrdinaryFeatures
            @cuda threads=default_num_threads2D blocks=ceil.(
                Int, get_cuda_count_tuple2d(r, i)) count_kernel_ordinary_get_candidate(
                    r.combs, 
                    r.refArray[i], 
                    r.cms.hash_coeffs, 
                    r.cms.sketch, 
                    r.selectedCombs[i],
                    min_count)
        elseif r.case == :Convolution
            @cuda threads=default_num_threads2D blocks=ceil.(
                Int, get_cuda_count_tuple2d(r, i)) count_kernel_conv_get_candidates(
                    r.combs, 
                    r.refArray[i], 
                    r.cms.hash_coeffs, 
                    r.cms.sketch, 
                    r.selectedCombs[i],
                    min_count)      
        else
            error("Unsupported case: $(r.case)")
        end
        CUDA.synchronize()
    end
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

end


export CountMinSketch, 
       Record

end
