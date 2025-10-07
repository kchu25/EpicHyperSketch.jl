# CPU implementations for both convolution and ordinary case counting
# These functions replicate the GPU kernel logic for CPU execution

using DataFrames

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
CPU version: Check if all filters/features in a combination are present (non-zero) in refArray.
"""
function is_combination_valid_cpu(combs, refArray, comb_col_ind, n)
    @inbounds for elem_idx in axes(combs, 1)
        comb_index_value = combs[elem_idx, comb_col_ind]
        # Guard against padding/invalid indices (0) or indices outside refArray
        if comb_index_value == 0
            return false
        end
        if comb_index_value > size(refArray, 1) || refArray[comb_index_value, FILTER_INDEX_COLUMN, n] == 0
            return false
        end
    end
    return true
end

"""
CPU version: Calculate hash index for convolution method, including position distances.
Returns -1 if filters overlap (invalid combination).
"""
function calculate_conv_hash_cpu(combs, refArray, hashCoefficients, comb_col_ind, sketch_row_ind, n, num_rows_combs, filter_len)
    sketch_col_index = Int32(0)
    @inbounds for elem_idx in 1:num_rows_combs
        comb_index_value = combs[elem_idx, comb_col_ind]
        # Guard against invalid comb index
        if comb_index_value == 0 || comb_index_value > size(refArray, 1)
            return Int32(-1)
        end
        filter_index = refArray[comb_index_value, FILTER_INDEX_COLUMN, n]
        hash_coeff = hashCoefficients[sketch_row_ind, 2*(elem_idx-1)+1]
        sketch_col_index += filter_index * hash_coeff

        if elem_idx < num_rows_combs
            next_comb_index_value = combs[elem_idx+1, comb_col_ind]
            # Guard against invalid next index
            if next_comb_index_value == 0 || next_comb_index_value > size(refArray, 1)
                return Int32(-1)
            end
            position1 = refArray[comb_index_value, POSITION_COLUMN, n]
            position2 = refArray[next_comb_index_value, POSITION_COLUMN, n]
            distance = position2 - position1 - filter_len
            
            if distance < 0  # overlapping filters
                return Int32(-1)  # signal invalid
            end
            
            sketch_col_index += hashCoefficients[sketch_row_ind, 2*elem_idx] * distance
        end
    end
    return sketch_col_index
end

"""
CPU version: Calculate hash index for ordinary (non-convolution) counting method.
"""
function calculate_ordinary_hash_cpu(combs, refArray, hashCoefficients, comb_col_ind, sketch_row_ind, n, num_rows_combs)
    sketch_col_index = Int32(0)
    @inbounds for elem_idx in 1:num_rows_combs
        comb_index_value = combs[elem_idx, comb_col_ind]
        filter_index = refArray[comb_index_value, FILTER_INDEX_COLUMN, n]
        hash_coeff = hashCoefficients[sketch_row_ind, elem_idx]
        sketch_col_index += filter_index * hash_coeff
    end
    return sketch_col_index
end

"""
CPU version: Execute counting on the sketch for convolution case.
"""
function count_conv_cpu!(r::Record, config::HyperSketchConfig)
    @info "Running CPU counting for convolution case..."
    
    for batch_idx = 1:num_batches(r)
        refArray = r.vecRefArray[batch_idx]
        combs = r.combs
        hashCoefficients = r.cms.hash_coeffs
        sketch = r.cms.sketch
        filter_len = r.filter_len
        
        num_cols_combs = size(combs, 2)
        num_rows_sketch = size(sketch, 1)
        num_cols_sketch = size(sketch, 2)
        num_counters = num_rows_sketch * num_cols_sketch
        batch_size = size(refArray, 3)
        
        # Iterate over all combinations, sketch rows, and batch elements
        for comb_col_ind in 1:num_cols_combs
            for sketch_row_ind in 1:num_rows_sketch
                for n in 1:batch_size
                    if is_combination_valid_cpu(combs, refArray, comb_col_ind, n)
                        sketch_col_index = calculate_conv_hash_cpu(combs, refArray, hashCoefficients, 
                                                                 comb_col_ind, sketch_row_ind, n, 
                                                                 size(combs, 1), filter_len)
                        if sketch_col_index != -1
                            final_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
                            sketch[sketch_row_ind, final_index] += 1
                        end
                    end
                end
            end
        end
    end
end

"""
CPU version: Execute counting on the sketch for ordinary case.
"""
function count_ordinary_cpu!(r::Record, config::HyperSketchConfig)
    @info "Running CPU counting for ordinary case..."
    
    for batch_idx = 1:num_batches(r)
        refArray = r.vecRefArray[batch_idx]
        combs = r.combs
        hashCoefficients = r.cms.hash_coeffs
        sketch = r.cms.sketch
        
        num_cols_combs = size(combs, 2)
        num_rows_sketch = size(sketch, 1)
        num_cols_sketch = size(sketch, 2)
        num_counters = num_rows_sketch * num_cols_sketch
        batch_size = size(refArray, 3)
        
        # Iterate over all combinations, sketch rows, and batch elements
        for comb_col_ind in 1:num_cols_combs
            for sketch_row_ind in 1:num_rows_sketch
                for n in 1:batch_size
                    if is_combination_valid_cpu(combs, refArray, comb_col_ind, n)
                        sketch_col_index = calculate_ordinary_hash_cpu(combs, refArray, hashCoefficients, 
                                                                     comb_col_ind, sketch_row_ind, n, 
                                                                     size(combs, 1))
                        final_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
                        sketch[sketch_row_ind, final_index] += 1
                    end
                end
            end
        end
    end
end

"""
CPU version: Identify combinations that meet minimum count threshold for convolution case.
"""
function make_selection_conv_cpu!(r::Record, config::HyperSketchConfig)
    @info "Running CPU selection for convolution case..."
    
    for batch_idx = 1:num_batches(r)
        refArray = r.vecRefArray[batch_idx]
        combs = r.combs
        hashCoefficients = r.cms.hash_coeffs
        sketch = r.cms.sketch
        selectedCombs = r.selectedCombs[batch_idx]
        filter_len = r.filter_len
        min_count = config.min_count
        
        num_cols_combs = size(combs, 2)
        num_rows_sketch = size(sketch, 1)
        num_cols_sketch = size(sketch, 2)
        num_counters = num_rows_sketch * num_cols_sketch
        batch_size = size(refArray, 3)
        
        # Iterate over combinations and batch elements (we only need one sketch row for selection)
        for comb_col_ind in 1:num_cols_combs
            for n in 1:batch_size
                if is_combination_valid_cpu(combs, refArray, comb_col_ind, n)
                    # Use first sketch row for selection decision
                    sketch_row_ind = 1
                    sketch_col_index = calculate_conv_hash_cpu(combs, refArray, hashCoefficients, 
                                                             comb_col_ind, sketch_row_ind, n, 
                                                             size(combs, 1), filter_len)
                    if sketch_col_index != -1
                        final_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
                        if sketch[sketch_row_ind, final_index] ≥ min_count
                            selectedCombs[comb_col_ind, n] = true
                        end
                    end
                end
            end
        end
    end
end

"""
CPU version: Identify combinations that meet minimum count threshold for ordinary case.
"""
function make_selection_ordinary_cpu!(r::Record, config::HyperSketchConfig)
    @info "Running CPU selection for ordinary case..."
    
    for batch_idx = 1:num_batches(r)
        refArray = r.vecRefArray[batch_idx]
        combs = r.combs
        hashCoefficients = r.cms.hash_coeffs
        sketch = r.cms.sketch
        selectedCombs = r.selectedCombs[batch_idx]
        min_count = config.min_count
        
        num_cols_combs = size(combs, 2)
        num_rows_sketch = size(sketch, 1)
        num_cols_sketch = size(sketch, 2)
        num_counters = num_rows_sketch * num_cols_sketch
        batch_size = size(refArray, 3)
        
        # Iterate over combinations and batch elements
        for comb_col_ind in 1:num_cols_combs
            for n in 1:batch_size
                if is_combination_valid_cpu(combs, refArray, comb_col_ind, n)
                    # Use first sketch row for selection decision
                    sketch_row_ind = 1
                    sketch_col_index = calculate_ordinary_hash_cpu(combs, refArray, hashCoefficients, 
                                                                 comb_col_ind, sketch_row_ind, n, 
                                                                 size(combs, 1))
                    final_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
                    if sketch[sketch_row_ind, final_index] ≥ min_count
                        selectedCombs[comb_col_ind, n] = true
                    end
                end
            end
        end
    end
end

"""
CPU version: Extract configurations for convolution case - filter IDs, distances, positions, data_index, contributions.
Returns matrices: motifs, distances, positions, data_index, contribs.
"""
function obtain_motifs_conv_cpu!(where_exceeds, combs, refArray, refArrayContrib, motifs_obtained, distances_obtained, 
                                  positions_obtained, data_index, contribs, filter_len, offset)
    for i in 1:length(where_exceeds)
        comb_idx, n = where_exceeds[i][1], where_exceeds[i][2]  # combination index, batch index
        K = size(combs, 1)
        
        # Store data_index (1-based, offset by batch)
        data_index[i] = offset + n
        
        # Calculate contribution as sum of all features in this combination
        total_contrib = 0.0f0
        @inbounds for k = 1:K
            comb_num = combs[k, comb_idx]
            total_contrib += refArrayContrib[comb_num, n]
        end
        contribs[i] = total_contrib
        
        # Extract motifs, distances, and positions
        @inbounds for k = 1:K
            comb_num = combs[k, comb_idx]
            # Store filter ID
            motifs_obtained[i, k] = refArray[comb_num, FILTER_INDEX_COLUMN, n]
            
            # Store distance to next filter (if not last)
            if k < K
                pos1 = refArray[comb_num, POSITION_COLUMN, n]
                pos2 = refArray[combs[k+1, comb_idx], POSITION_COLUMN, n]
                distances_obtained[i, k] = pos2 - pos1 - filter_len
            end
        end
        
        # Store start and end positions
        positions_obtained[i, 1] = refArray[combs[1, comb_idx], POSITION_COLUMN, n]  # start
        positions_obtained[i, 2] = refArray[combs[K, comb_idx], POSITION_COLUMN, n] + filter_len - 1  # end
    end
end

"""
CPU version: Extract configurations for ordinary case - feature IDs, data_index, contributions.
Returns matrices: motifs, data_index, contribs.
"""
function obtain_motifs_ordinary_cpu!(where_exceeds, combs, refArray, refArrayContrib, motifs_obtained, data_index, contribs, offset)
    for i in 1:length(where_exceeds)
        comb_idx, n = where_exceeds[i][1], where_exceeds[i][2]  # combination index, batch index
        K = size(combs, 1)
        
        # Store data_index (1-based, offset by batch)
        data_index[i] = offset + n
        
        # Calculate contribution as sum of all features in this combination
        total_contrib = 0.0f0
        @inbounds for k = 1:K
            comb_num = combs[k, comb_idx]
            total_contrib += refArrayContrib[comb_num, n]
        end
        contribs[i] = total_contrib
        
        # Extract feature IDs
        @inbounds for k = 1:K
            motifs_obtained[i, k] = refArray[combs[k, comb_idx], FILTER_INDEX_COLUMN, n]
        end
    end
end

"""
CPU version: Main counting function dispatcher.
"""
function count_cpu!(r::Record, config::HyperSketchConfig)
    if r.case == :OrdinaryFeatures
        count_ordinary_cpu!(r, config)
    elseif r.case == :Convolution
        @assert r.filter_len !== nothing "Convolution case requires a numeric `filter_len` (got `nothing`)."
        count_conv_cpu!(r, config)
    else
        error("Unsupported case: $(r.case)")
    end
end

"""
CPU version: Main selection function dispatcher.
"""
function make_selection_cpu!(r::Record, config::HyperSketchConfig)
    if r.case == :OrdinaryFeatures
        make_selection_ordinary_cpu!(r, config)
    elseif r.case == :Convolution
        @assert r.filter_len !== nothing "Convolution case requires a numeric `filter_len` (got `nothing`)."
        make_selection_conv_cpu!(r, config)
    else
        error("Unsupported case: $(r.case)")
    end
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
    contrib_vec = Vector{Vector{Float32}}()
    distances_vec = r.case == :Convolution ? Vector{Matrix{IntType}}() : nothing
    positions_vec = r.case == :Convolution ? Vector{Matrix{IntType}}() : nothing
    
    offset = 0  # Track the data_index offset across batches
    
    for batch_idx = 1:num_batches(r)
        where_exceeds = findall(r.selectedCombs[batch_idx] .== true)
        batch_size = size(r.vecRefArray[batch_idx], 3)
        
        if !isempty(where_exceeds)
            n_configs = length(where_exceeds)
            
            if r.case == :OrdinaryFeatures
                # Allocate arrays for ordinary case
                motifs = Matrix{IntType}(undef, n_configs, r.motif_size)
                data_index = Vector{IntType}(undef, n_configs)
                contribs = Vector{Float32}(undef, n_configs)
                
                # Extract configurations
                obtain_motifs_ordinary_cpu!(where_exceeds, r.combs, r.vecRefArray[batch_idx],
                                           r.vecRefArrayContrib[batch_idx],
                                           motifs, data_index, contribs, offset)
                
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
                contribs = Vector{Float32}(undef, n_configs)
                
                # Extract configurations
                obtain_motifs_conv_cpu!(where_exceeds, r.combs, r.vecRefArray[batch_idx],
                                       r.vecRefArrayContrib[batch_idx],
                                       motifs, distances, positions, data_index, contribs,
                                       r.filter_len, offset)
                
                # Store results
                push!(motifs_vec, motifs)
                push!(distances_vec, distances)
                push!(positions_vec, positions)
                push!(data_idx_vec, data_index)
                push!(contrib_vec, contribs)
            end
        end
        
        offset += batch_size  # Increment by number of sequences in batch
    end
    
    # If no configurations found, return empty DataFrame with proper structure
    if isempty(motifs_vec)
        if r.case == :OrdinaryFeatures
            motif_cols = [Symbol("m$i") for i in 1:r.motif_size]
            col_names = [motif_cols..., :data_index, :contribution]
            return DataFrame([name => IntType[] for name in motif_cols]..., 
                           :data_index => IntType[], :contribution => Float32[])
        else  # Convolution
            motif_cols = [Symbol("m$i") for i in 1:r.motif_size]
            distance_cols = [Symbol("d$(i)$(i+1)") for i in 1:(r.motif_size-1)]
            position_cols = [:start, :end]
            col_names = [motif_cols..., distance_cols..., position_cols..., :data_index, :contribution]
            return DataFrame([name => IntType[] for name in motif_cols]...,
                           [name => IntType[] for name in distance_cols]...,
                           [name => IntType[] for name in position_cols]...,
                           :data_index => IntType[], :contribution => Float32[])
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