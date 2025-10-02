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


function _obtain_enriched_configurations_(r::Record)
    #= get the configurations; (i.e. where 
        (combination, seq) in the placeholder 
        from the sketch exceed min_count) =#
    enriched_configs = Vector{Set{Tuple}}(undef, num_batches(r))
    for i = 1:num_batches(r)
        # @info "obtaining enriched configurations for batch $i"
        where_exceeds = findall(r.selectedCombs[i] .== true)
        # @info "grid : $(where_exceeds)"
        
        configs = CuMatrix{int_type}(IntType, (length(where_exceeds), config_size(r)));

        if length(where_exceeds) == 0
            enriched_configs[i] = Set{Vector{IntType}}() 
        else
            @assert r.use_cuda "obtain_enriched_configurations currently only supports use_cuda=true"

            if r.case == :OrdinaryFeatures
                @cuda threads=default_num_threads1D blocks=ceil(IntType, length(where_exceeds)) obtain_configs_ordinary!(
                    where_exceeds, r.combs, r.refArray[i], configs)
            elseif r.case == :Convolution
                @cuda threads=default_num_threads1D blocks=ceil(IntType, length(where_exceeds)) obtain_configs_conv!(
                    where_exceeds, r.combs, r.refArray[i], configs, r.filter_len)
            else
                error("Unsupported case: $(r.case)")
            end

            enriched_configs[i] = map(x->Tuple(x), eachrow(Array(configs))) |> Set
        end
    end

    # TODO: figure out the enriched_configs type
    # TODO: refactor this later

    set_here = reduce(union, enriched_configs)
    return set_here
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
