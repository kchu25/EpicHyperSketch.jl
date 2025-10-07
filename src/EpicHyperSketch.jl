module EpicHyperSketch

using CUDA
using Combinatorics
using Random
using DataFrames
#

const IntType = Int32 # RTX 3090 has dedicated INT32 execution units in each SM 
const FloatType = Float32
# const PrimeNumber = IntType(50000101)  # A large prime number for hashing
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
const POSITION_COLUMN = 2     # Position in sequence


# Number of columns (2nd dimension) in refArray based on case
const refArraysDim = Dict(
    :OrdinaryFeatures => 1,
    :Convolution => 2
)

include("types.jl")
include("errors.jl") 
include("config.jl")
include("performance.jl")
include("sketch.jl")
include("record.jl")
include("count_gpu.jl")
include("count_cpu.jl")


# Helper function to dispatch kernel based on case
function _launch_count_kernel!(r::Record, batch_idx::Int, config::HyperSketchConfig)
    common_args = (r.combs, r.vecRefArray[batch_idx], r.cms.hash_coeffs, r.cms.sketch)
    threads = config.threads_3d
    blocks = ceil.(Int, get_cuda_count_tuple3d(r, batch_idx))
    
    if r.case == :OrdinaryFeatures
        @cuda threads=threads blocks=blocks count_kernel_ordinary(common_args...)
    elseif r.case == :Convolution
        @assert r.filter_len !== nothing "Convolution case requires a numeric `filter_len` (got `nothing`)."
        @cuda threads=threads blocks=blocks count_kernel_conv(common_args..., r.filter_len)
    else
        error("Unsupported case: $(r.case)")
    end
end

# Helper function to dispatch candidate selection kernel based on case
function _launch_selection_kernel!(r::Record, batch_idx::Int, config::HyperSketchConfig)
    threads = config.threads_2d
    num_combs = size(r.combs, 2)
    batch_size = size(r.vecRefArray[batch_idx], 3)
    blocks = (cld(num_combs, threads[1]), cld(batch_size, threads[2]))
    common_args = (r.combs, r.vecRefArray[batch_idx], r.cms.hash_coeffs, 
        r.cms.sketch, r.selectedCombs[batch_idx], config.min_count)

    if r.case == :OrdinaryFeatures
        @cuda threads=threads blocks=blocks count_kernel_ordinary_get_candidate(common_args...)
    elseif r.case == :Convolution
        @assert r.filter_len !== nothing "Convolution case requires a numeric `filter_len` (got `nothing`)."
        @cuda threads=threads blocks=blocks count_kernel_conv_get_candidates(common_args..., r.filter_len)
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
        _launch_selection_kernel!(r, batch_idx, config)
        CUDA.synchronize()
    end
end






# Helper to initialize result arrays for config extraction
function _initialize_result_arrays(where_exceeds_vec, motif_size, case::Symbol)
    n_batches = length(where_exceeds_vec)
    
    # Common arrays for both cases
    motifs_obtained_vec = [CUDA.zeros(IntType, (length(w), motif_size)) for w in where_exceeds_vec]
    data_index_vec = [CUDA.zeros(IntType, length(w)) for w in where_exceeds_vec]
    contribution_vec = [CUDA.zeros(FloatType, length(w)) for w in where_exceeds_vec]
    
    # Case-specific arrays
    if case == :Convolution
        distances_vec = [CUDA.zeros(IntType, (length(w), motif_size - 1)) for w in where_exceeds_vec]
        positions_vec = [CUDA.zeros(IntType, (length(w), 2)) for w in where_exceeds_vec]
        return motifs_obtained_vec, data_index_vec, contribution_vec, distances_vec, positions_vec
    else
        return motifs_obtained_vec, data_index_vec, contribution_vec, nothing, nothing
    end
end

# Helper to launch ordinary feature extraction kernels
function _launch_ordinary_extraction!(r::Record, where_exceeds_vec, motifs_vec, data_idx_vec, contrib_vec, threads)
    offset = IntType(0)
    
    for batch_idx = 1:num_batches(r)
        n_items = length(where_exceeds_vec[batch_idx])
        batch_size = size(r.vecRefArray[batch_idx], 3)  # Number of sequences in this batch
        
        if n_items > 0  # Only launch kernel if there are items to process
            blocks = cld(n_items, threads)
            @cuda threads=threads blocks=blocks obtain_motifs_ordinary!(
                where_exceeds_vec[batch_idx], 
                r.combs, 
                r.vecRefArray[batch_idx], 
                r.vecRefArrayContrib[batch_idx],
                motifs_vec[batch_idx],
                data_idx_vec[batch_idx],
                contrib_vec[batch_idx],
                offset
            )
        end
        
        offset += batch_size  # Increment by number of sequences, not number of enriched motifs
    end
end

# Helper to launch convolution extraction kernels
function _launch_convolution_extraction!(r::Record, where_exceeds_vec, motifs_vec, distances_vec, 
                                        data_idx_vec, positions_vec, contrib_vec, threads)
    offset = IntType(0)
    
    for batch_idx = 1:num_batches(r)
        n_items = length(where_exceeds_vec[batch_idx])
        batch_size = size(r.vecRefArray[batch_idx], 3)  # Number of sequences in this batch
        
        if n_items > 0  # Only launch kernel if there are items to process
            blocks = cld(n_items, threads)
            @cuda threads=threads blocks=blocks obtain_motifs_conv!(
                where_exceeds_vec[batch_idx], 
                r.combs, 
                r.vecRefArray[batch_idx], 
                r.vecRefArrayContrib[batch_idx],
                motifs_vec[batch_idx], 
                distances_vec[batch_idx],
                data_idx_vec[batch_idx],
                positions_vec[batch_idx],
                contrib_vec[batch_idx],
                r.filter_len,
                offset
            )
        end
        
        offset += batch_size  # Increment by number of sequences, not number of enriched motifs
    end
end

# Helper to create DataFrame for ordinary features
function _create_ordinary_dataframe(motifs, data_index, contribs, motif_size)
    motif_cols = [Symbol("m$i") for i in 1:motif_size]
    col_names = [motif_cols..., :data_index, :contribution]
    
    df_data = hcat(Array(motifs), Array(data_index), Array(contribs))
    return DataFrame(df_data, col_names)
end

# Helper to create DataFrame for convolution features
function _create_convolution_dataframe(motifs, distances, positions, data_index, contribs, motif_size)
    motif_cols = [Symbol("m$i") for i in 1:motif_size]
    distance_cols = [Symbol("d$(i)$(i+1)") for i in 1:(motif_size-1)]
    position_cols = [:start, :end]
    col_names = [motif_cols..., distance_cols..., position_cols..., :data_index, :contribution]
    
    df_data = hcat(Array(motifs), Array(distances), Array(positions), Array(data_index), Array(contribs))
    return DataFrame(df_data, col_names)
end

# Main function to extract and format enriched configurations
function _launch_config_kernels(r::Record, where_exceeds_vec, config::HyperSketchConfig)
    threads = config.threads_1d
    
    # Initialize result arrays
    motifs_vec, data_idx_vec, contrib_vec, distances_vec, positions_vec = 
        _initialize_result_arrays(where_exceeds_vec, r.motif_size, r.case)
    
    # Launch appropriate kernels based on case
    if r.case == :OrdinaryFeatures
        _launch_ordinary_extraction!(r, where_exceeds_vec, motifs_vec, data_idx_vec, contrib_vec, threads)
        
        # Combine results from all batches
        motifs = reduce(vcat, motifs_vec)
        data_index = reduce(vcat, data_idx_vec)
        contribs = reduce(vcat, contrib_vec)
        
        # Create and return DataFrame
        return _create_ordinary_dataframe(motifs, data_index, contribs, r.motif_size)
        
    elseif r.case == :Convolution
        @assert r.filter_len !== nothing "Convolution case requires filter_len"
        
        _launch_convolution_extraction!(r, where_exceeds_vec, motifs_vec, distances_vec,
                                       data_idx_vec, positions_vec, contrib_vec, threads)
        
        # Combine results from all batches
        motifs = reduce(vcat, motifs_vec)
        distances = reduce(vcat, distances_vec)
        positions = reduce(vcat, positions_vec)
        data_index = reduce(vcat, data_idx_vec)
        contribs = reduce(vcat, contrib_vec)
        
        # Create and return DataFrame
        return _create_convolution_dataframe(motifs, distances, positions, data_index, contribs, r.motif_size)
        
    else
        error("Unsupported case: $(r.case)")
    end
end

function _obtain_enriched_configurations_(r::Record, config::HyperSketchConfig)
    """Extract configurations where combinations exceed minimum count threshold."""
    @assert config.use_cuda "obtain_enriched_configurations currently only supports use_cuda=true"
    
    # enriched_motifs = Vector{Set{Tuple}}(undef, num_batches(r))
    where_exceeds_vec = Vector{Vector{CartesianIndex{2}}}(undef, num_batches(r))
    for batch_idx = 1:num_batches(r)
        where_exceeds = findall(r.selectedCombs[batch_idx] .== true)
        where_exceeds_vec[batch_idx] = where_exceeds
        # if isempty(where_exceeds)
        #     enriched_motifs[batch_idx] = Set{Vector{IntType}}()
        # else
        #     motifs_obtained = CuMatrix{IntType}(undef, length(where_exceeds), actual_motif_size(r))
        #     _launch_config_kernel!(r, batch_idx, where_exceeds, motifs_obtained, config)
        #     enriched_motifs[batch_idx] = Set(map(Tuple, eachrow(Array(motifs_obtained))))
        # end
    end

    result_df = _launch_config_kernels(r, where_exceeds_vec, config)

    return result_df
end


function obtain_enriched_configurations(
    activation_dict::ActivationDict;
    motif_size::Integer=3,
    filter_len::Union{Integer,Nothing}=nothing,
    min_count::Integer=1, 
    seed::Union{Integer, Nothing}=1,
    config::HyperSketchConfig=default_config(min_count=min_count, seed=seed)
)
    # Validation
    validate_activation_dict(activation_dict)
    validate_motif_size(motif_size)
    check_cuda_requirements(config.use_cuda)
    
    # Create record with configuration
    @info "Constructing Record..."
    r = Record(activation_dict, motif_size; 
               batch_size=config.batch_size,
               use_cuda=config.use_cuda, 
               filter_len=filter_len,
               seed=config.seed)

    @info "Starting to count..."
    # Execute pipeline
    count!(r, config)
    @info "Counting completed. Starting selection..."
    make_selection!(r, config)

    # Debug selection results
    # @info "Selection results for first batch:"
    # if !isempty(r.selectedCombs)
    #     selected_indices = findall(Array(r.selectedCombs[1]) .== true)
    #     println("Selected combination indices: ", selected_indices)
        
    #     # Show which actual combinations were selected
    #     for idx in selected_indices[1:min(5, end)]
    #         comb = r.combs[:, idx[1]]
    #         println("Selected combination: ", comb)
    #     end
    # end

    motifs = _obtain_enriched_configurations_(r, config)

    return motifs
end

export CountMinSketch, 
       Record,
       HyperSketchConfig,
       default_config,
       obtain_enriched_configurations,
       # CPU versions
       obtain_enriched_configurations_cpu,
       count_cpu!,
       make_selection_cpu!,
       # Errors
       HyperSketchError,
       InvalidConfigurationError,
       CUDAError

end
