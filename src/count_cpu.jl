# CPU implementations for both convolution and ordinary case counting
# These functions replicate the GPU kernel logic for CPU execution

using DataFrames

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



