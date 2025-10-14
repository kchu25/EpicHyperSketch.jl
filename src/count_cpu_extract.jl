
# Helper to create DataFrame for ordinary features (CPU version)
function _create_ordinary_dataframe_cpu(motifs, data_index, contribs, motif_size)
    motif_cols = [Symbol("m$i") for i in 1:motif_size]
    
    # Create DataFrame column by column to preserve types
    df = DataFrame()
    for i in 1:motif_size
        df[!, motif_cols[i]] = motifs[:, i]
    end
    df[!, :data_index] = data_index
    df[!, :contribution] = contribs
    
    return df
end

# Helper to create DataFrame for convolution features (CPU version)
function _create_convolution_dataframe_cpu(motifs, distances, positions, data_index, contribs, motif_size)
    motif_cols = [Symbol("m$i") for i in 1:motif_size]
    distance_cols = [Symbol("d$(i)$(i+1)") for i in 1:(motif_size-1)]
    position_cols = [:start, :end]
    
    # Create DataFrame column by column to preserve types
    df = DataFrame()
    for i in 1:motif_size
        df[!, motif_cols[i]] = motifs[:, i]
    end
    for i in 1:(motif_size-1)
        df[!, distance_cols[i]] = distances[:, i]
    end
    df[!, :start] = positions[:, 1]
    df[!, :end] = positions[:, 2]
    df[!, :data_index] = data_index
    df[!, :contribution] = contribs
    
    return df
end


"""
CPU version: Extract configurations where combinations exceed minimum count threshold.
Returns a DataFrame with the same structure as the GPU version.
"""
function _obtain_enriched_configurations_cpu_(r::Record, config::HyperSketchConfig)
    @info "Running CPU configuration extraction..."
    
    # Collect results from all batches
    motifs_vec = Vector{Matrix{IntType}}()
    data_idx_vec = Vector{Vector{IntType}}()
    contrib_vec = Vector{Vector{FloatType}}()
    distances_vec = r.case == :Convolution ? Vector{Matrix{IntType}}() : nothing
    positions_vec = r.case == :Convolution ? Vector{Matrix{IntType}}() : nothing
    
    for batch_idx = 1:num_batches(r)
        where_exceeds = findall(r.selectedCombs[batch_idx] .== true)
        
        if !isempty(where_exceeds)
            n_configs = length(where_exceeds)
            
            if r.case == :OrdinaryFeatures
                # Allocate arrays for ordinary case
                motifs = Matrix{IntType}(undef, n_configs, r.motif_size)
                data_index = Vector{IntType}(undef, n_configs)
                contribs = Vector{FloatType}(undef, n_configs)
                
                # Extract configurations
                obtain_motifs_ordinary_cpu!(where_exceeds, r.combs, r.vecRefArray[batch_idx],
                                           r.vecRefArrayContrib[batch_idx],
                                           motifs, data_index, contribs)
                
                # Store results
                push!(motifs_vec, motifs)
                push!(data_idx_vec, data_index)
                push!(contrib_vec, contribs)
                
            elseif r.case == :Convolution
                @assert r.filter_len !== nothing "Convolution case requires filter_len"
                
                # Allocate arrays for convolution case
                motifs = Matrix{IntType}(undef, n_configs, r.motif_size)
                distances = Matrix{IntType}(undef, n_configs, r.motif_size - 1)
                positions = Matrix{IntType}(undef, n_configs, 2)  # start, end
                data_index = Vector{IntType}(undef, n_configs)
                contribs = Vector{FloatType}(undef, n_configs)
                
                # Extract configurations
                obtain_motifs_conv_cpu!(where_exceeds, r.combs, r.vecRefArray[batch_idx],
                                       r.vecRefArrayContrib[batch_idx],
                                       motifs, distances, positions, data_index, contribs,
                                       r.filter_len)
                
                # Store results
                push!(motifs_vec, motifs)
                push!(distances_vec, distances)
                push!(positions_vec, positions)
                push!(data_idx_vec, data_index)
                push!(contrib_vec, contribs)
            end
        end
    end
    
    # If no configurations found, return empty DataFrame with proper structure
    if isempty(motifs_vec)
        if r.case == :OrdinaryFeatures
            motif_cols = [Symbol("m$i") for i in 1:r.motif_size]
            # col_names = [motif_cols..., :data_pt_index, :contribution]
            return DataFrame([name => IntType[] for name in motif_cols]..., 
                           :data_pt_index => IntType[], :contribution => FloatType[])
        else  # Convolution
            motif_cols = [Symbol("m$i") for i in 1:r.motif_size]
            distance_cols = [Symbol("d$(i)$(i+1)") for i in 1:(r.motif_size-1)]
            position_cols = [:start, :end]
            # col_names = [motif_cols..., distance_cols..., position_cols..., :data_pt_index, :contribution]
            return DataFrame([name => IntType[] for name in motif_cols]...,
                           [name => IntType[] for name in distance_cols]...,
                           [name => IntType[] for name in position_cols]...,
                           :data_pt_index => IntType[], :contribution => FloatType[])
        end
    end
    
    # Combine results from all batches and create DataFrame
    if r.case == :OrdinaryFeatures
        motifs = reduce(vcat, motifs_vec)
        data_index = reduce(vcat, data_idx_vec)
        contribs = reduce(vcat, contrib_vec)
        
        return _create_ordinary_dataframe_cpu(motifs, data_index, contribs, r.motif_size)
        
    else  # Convolution
        motifs = reduce(vcat, motifs_vec)
        distances = reduce(vcat, distances_vec)
        positions = reduce(vcat, positions_vec)
        data_index = reduce(vcat, data_idx_vec)
        contribs = reduce(vcat, contrib_vec)
        
        return _create_convolution_dataframe_cpu(motifs, distances, positions, data_index, contribs, r.motif_size)
    end
end

"""
CPU version: Main function to obtain enriched configurations.
"""
function obtain_enriched_configurations_cpu(
    activation_dict::ActivationDict;
    motif_size::Integer=3,
    filter_len::Union{Integer,Nothing}=nothing,
    min_count::Integer=1, 
    seed::Union{Integer, Nothing}=1,
    config::HyperSketchConfig=default_config(min_count=min_count,
        use_cuda=false, seed=seed)
)
    # Validation
    validate_activation_dict(activation_dict)
    validate_motif_size(motif_size)
    
    # Override CUDA setting for CPU version
    config = HyperSketchConfig(
        delta=config.delta,
        epsilon=config.epsilon,
        min_count=config.min_count,
        batch_size=config.batch_size,
        use_cuda=false,  # Force CPU
        threads_1d=config.threads_1d,
        threads_2d=config.threads_2d,
        threads_3d=config.threads_3d,
        seed=config.seed  # Preserve seed for reproducibility
    )
    
    @info "Constructing Record (CPU mode)..."
    r = Record(activation_dict, motif_size; 
               batch_size=config.batch_size,
               use_cuda=false,  # Force CPU arrays
               filter_len=filter_len,
               seed=config.seed)

    @info "Starting CPU counting..."
    count_cpu!(r, config)
    
    @info "CPU counting completed. Starting selection..."
    make_selection_cpu!(r, config)
    
    @info "Selection completed. Extracting configurations..."
    return _obtain_enriched_configurations_cpu_(r, config)
end