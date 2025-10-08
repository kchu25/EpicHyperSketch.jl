"""
Check if all filters/features in a combination are present (non-zero) in refArray.
"""
function is_combination_valid(combs, refArray, comb_col_ind, n)
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
Calculate hash index for ordinary (non-convolution) counting method.
"""
function calculate_ordinary_hash(combs, refArray, hashCoefficients, comb_col_ind, sketch_row_ind, n, num_rows_combs)
    sketch_col_index = IntType(0)
    @inbounds for elem_idx in 1:num_rows_combs
        comb_index_value = combs[elem_idx, comb_col_ind]
        filter_index = refArray[comb_index_value, FILTER_INDEX_COLUMN, n]  # Get filter/feature ID  # Get filter/feature ID
        hash_coeff = hashCoefficients[sketch_row_ind, elem_idx]
        sketch_col_index += filter_index * hash_coeff
    end
    return sketch_col_index
end

"""
Calculate hash index for convolution method, including position distances.
Returns -1 if filters overlap (invalid combination).
"""
function calculate_conv_hash(combs, refArray, hashCoefficients, comb_col_ind, sketch_row_ind, n, num_rows_combs, filter_len)
    # Extract combination indices and check validity
    @inbounds for elem_idx in 1:num_rows_combs
        comb_index_value = combs[elem_idx, comb_col_ind]
        if comb_index_value == 0 || comb_index_value > size(refArray,1)
            return IntType(-1)
        end
    end
    
    # Check for overlaps in original order
    @inbounds for i in 1:(num_rows_combs-1)
        comb_idx1 = combs[i, comb_col_ind]
        comb_idx2 = combs[i+1, comb_col_ind]
        pos1 = refArray[comb_idx1, POSITION_COLUMN, n]
        pos2 = refArray[comb_idx2, POSITION_COLUMN, n]
        distance = pos2 - pos1 - filter_len
        if distance < 0  # overlapping filters
            return IntType(-1)  # signal invalid
        end
    end
    
    # Calculate hash using original order for both filters and distances
    sketch_col_index = IntType(0)
    @inbounds for elem_idx in 1:num_rows_combs
        comb_idx = combs[elem_idx, comb_col_ind]
        filter_index = refArray[comb_idx, FILTER_INDEX_COLUMN, n]
        hash_coeff = hashCoefficients[sketch_row_ind, 2*(elem_idx-1)+1]
        sketch_col_index += filter_index * hash_coeff

        if elem_idx < num_rows_combs
            next_comb_idx = combs[elem_idx+1, comb_col_ind]
            position1 = refArray[comb_idx, POSITION_COLUMN, n]
            position2 = refArray[next_comb_idx, POSITION_COLUMN, n]
            distance = position2 - position1 - filter_len
            
            sketch_col_index += hashCoefficients[sketch_row_ind, 2*elem_idx] * distance
        end
    end
    return sketch_col_index
end

"""
CUDA kernel for convolution-based counting with position-aware hashing.
Works with filter/feature combinations, skips when filters overlap.
"""
function count_kernel_conv(combs, refArray, hashCoefficients, sketch, filter_len)
    comb_col_ind, sketch_row_ind, n, num_counters, num_cols_sketch, in_bounds = 
        _kernel_setup_and_bounds_check(combs, refArray, sketch)
    
    if in_bounds && is_combination_valid(combs, refArray, comb_col_ind, n)
        sketch_col_index = calculate_conv_hash(combs, refArray, hashCoefficients, 
                                             comb_col_ind, sketch_row_ind, n, 
                                             size(combs, 1), filter_len)
        if sketch_col_index != -1
            _process_sketch_update!(sketch, sketch_row_ind, sketch_col_index, num_counters, num_cols_sketch)
        end
    end
    return nothing
end

"""
CUDA kernel for ordinary counting of filter/feature combinations without position constraints.
"""
function count_kernel_ordinary(combs, refArray, hashCoefficients, sketch)
    comb_col_ind, sketch_row_ind, n, num_counters, num_cols_sketch, in_bounds = 
        _kernel_setup_and_bounds_check(combs, refArray, sketch)
    
    if in_bounds && is_combination_valid(combs, refArray, comb_col_ind, n)
        sketch_col_index = calculate_ordinary_hash(combs, refArray, hashCoefficients, 
                                                 comb_col_ind, sketch_row_ind, n, 
                                                 size(combs, 1))
        _process_sketch_update!(sketch, sketch_row_ind, sketch_col_index, num_counters, num_cols_sketch)
    end
    return nothing
end

# Common kernel setup and bounds checking
function _kernel_setup_and_bounds_check(combs, refArray, sketch)
    comb_col_ind = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sketch_row_ind = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    _, num_cols_combs = size(combs)
    num_rows_sketch, num_cols_sketch = size(sketch)
    num_counters = num_rows_sketch * num_cols_sketch
    
    in_bounds = (comb_col_ind ≤ num_cols_combs && 
                 sketch_row_ind ≤ num_rows_sketch && 
                 n ≤ size(refArray, 3))
    
    return (comb_col_ind, sketch_row_ind, n, num_counters, num_cols_sketch, in_bounds)
end

# Common sketch index calculation and update logic
function _process_sketch_update!(sketch, sketch_row_ind, sketch_col_index, num_counters, num_cols_sketch)
    final_index = mod(mod(sketch_col_index, num_counters), num_cols_sketch) + 1
    CUDA.@atomic sketch[sketch_row_ind, final_index] += 1
end

# Count-Min Sketch selection: compute minimum across all hash functions
function _compute_cms_minimum_count(combs, refArray, hashCoefficients, sketch, comb_col_ind, n, 
                                   num_counters, num_cols_sketch, hash_func)
    min_count = typemax(IntType)
    num_hash_functions = size(sketch, 1)
    
    @inbounds for row = 1:num_hash_functions
        sketch_col_index = hash_func(combs, refArray, hashCoefficients, comb_col_ind, row, n)
        if sketch_col_index != -1  # valid hash (for conv case)
            final_index = mod(mod(sketch_col_index, num_counters), num_cols_sketch) + 1
            count = sketch[row, final_index]
            min_count = min(min_count, count)
        else
            return IntType(-1)  # invalid combination
        end
    end
    return min_count
end

"""
CUDA kernel for convolution-based candidate selection with position-aware hashing.
Computes minimum across all hash functions for proper Count-Min Sketch behavior.
"""
function count_kernel_conv_get_candidates(combs, refArray, hashCoefficients, 
        sketch, selectedCombs, min_count, filter_len)
    comb_col_ind = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    num_cols_combs = size(combs, 2)
    num_counters = size(sketch, 1) * size(sketch, 2)
    num_cols_sketch = size(sketch, 2)
    
    if comb_col_ind ≤ num_cols_combs && n ≤ size(refArray, 3) && is_combination_valid(combs, refArray, comb_col_ind, n)
        # Define hash function for convolution case
        hash_func = (combs, refArray, hashCoefficients, comb_col_ind, row, n) -> begin
            return calculate_conv_hash(combs, refArray, hashCoefficients, comb_col_ind, row, n, 
                                     size(combs, 1), filter_len)
        end
        
        cms_count = _compute_cms_minimum_count(combs, refArray, hashCoefficients, sketch, 
                                              comb_col_ind, n, num_counters, num_cols_sketch, hash_func)
        if cms_count ≥ min_count
            #  @cuprintf("cms_count=%d, min_count=%d\n", cms_count, min_count)
            #  @cuprintf("min_count=%d\n", min_count)
            #  @cuprintf(cms_count)
            selectedCombs[comb_col_ind, n] = true
        end
    end
    return nothing
end

"""
CUDA kernel for ordinary candidate selection without position constraints.
Computes minimum across all hash functions for proper Count-Min Sketch behavior.
"""
function count_kernel_ordinary_get_candidate(combs, refArray, hashCoefficients, sketch, selectedCombs, min_count)
    comb_col_ind = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    num_cols_combs = size(combs, 2)
    num_counters = size(sketch, 1) * size(sketch, 2)
    num_cols_sketch = size(sketch, 2)
    
    if comb_col_ind ≤ num_cols_combs && n ≤ size(refArray, 3) && is_combination_valid(combs, refArray, comb_col_ind, n)
        # Define hash function for ordinary case
        hash_func = (combs, refArray, hashCoefficients, comb_col_ind, row, n) -> begin
            return calculate_ordinary_hash(combs, refArray, hashCoefficients, comb_col_ind, row, n, 
                                         size(combs, 1))
        end
        
        cms_count = _compute_cms_minimum_count(combs, refArray, hashCoefficients, sketch, 
                                              comb_col_ind, n, num_counters, num_cols_sketch, hash_func)
        if cms_count ≥ min_count
            selectedCombs[comb_col_ind, n] = true
        end
    end
    return nothing
end

# Common configuration extraction setup
function _config_kernel_setup(CindsVec)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    I = length(CindsVec)
    in_bounds = i ≤ I
    
    if in_bounds
        j, n = CindsVec[i][1], CindsVec[i][2]  # j-th combination, n-th sequence
        return (i, j, n, true)
    else
        return (i, 0, 0, false)
    end
end

"""
Extract configurations for convolution case: filter IDs and inter-filter distances.
"""
function obtain_motifs_conv!(CindsVec, combs, refArray, refArrayContrib, motifs_obtained, distances, data_index, positions, contribs, filter_len)
    i, j, n, in_bounds = _config_kernel_setup(CindsVec)
    
    if in_bounds
        K = size(combs, 1)
        contrib = FloatType(0)
        
        last_comb_num = IntType(0)  # Track last combination number for data_index
        @inbounds for k = 1:K
            comb_num = combs[k, j]
            last_comb_num = comb_num  # Save for use after loop
            motifs_obtained[i, k] = refArray[comb_num, FILTER_INDEX_COLUMN, n]
            # Store distance to next filter (if not last)
            if k < K
                pos1 = refArray[comb_num, POSITION_COLUMN, n]
                pos2 = refArray[combs[k+1, j], POSITION_COLUMN, n]
                distances[i, k] = pos2 - pos1 - filter_len
            end
            # store the position of the start and the end of the motif
            if k == 1
                positions[i, 1] = refArray[comb_num, POSITION_COLUMN, n]
            elseif k == K
                positions[i, 2] = refArray[comb_num, POSITION_COLUMN, n] + filter_len - 1
            end
            contrib += refArrayContrib[comb_num, n]
        end
        data_index[i] = refArray[last_comb_num, DATA_PT_INDEX_COLUMN, n]
        contribs[i] = contrib
    end
    return nothing
end

"""
Extract configurations for ordinary case: only filter/feature IDs.
"""
function obtain_motifs_ordinary!(CindsVec, combs, refArray, refArrayContrib, motifs_obtained, data_index, contribs)
    i, j, n, in_bounds = _config_kernel_setup(CindsVec)
    
    if in_bounds
        K = size(combs, 1)
        contrib = FloatType(0)
        last_comb_num = IntType(0)  # Track last combination number for data_index
        @inbounds for k = 1:K
            comb_num = combs[k, j]
            last_comb_num = comb_num  # Save for use after loop
            motifs_obtained[i, k] = refArray[comb_num, FILTER_INDEX_COLUMN, n]
            contrib += refArrayContrib[comb_num, n]
        end
        data_index[i] = refArray[last_comb_num, DATA_PT_INDEX_COLUMN, n]
        contribs[i] = contrib
    end
    return nothing
end


"""
Data Structure Specifications:

refArray: (max_num_active x 2 x batch_size) 3D array
    - Rows: number of active filters/features for a sequence
    - 2 Columns: [filter/feature ID, position in sequence]
    - Depth: batch size (number of sequences in current batch)

hashCoefficients: (num_hash_functions x "motif_size") matrix
    - Rows: number of hash functions (equals sketch height)
    - Columns: motif_size (ordinary) or 2 x motif_size - 1 (convolution)

combs: (num_motif_elements x num_combinations) matrix  
    - Rows: motif_size (ordinary) or ⌈motif_size/2⌉ (convolution)
    - Columns: total number of filter/feature combinations

selectedCombs: (num_combinations x min(batch_size, num_sequences)) boolean matrix
    - Rows: total number of filter/feature combinations
    - Columns: batch size (number of sequences in current batch)

Parameters:
    - motif_size: number of filters/features per motif
    - n: batch index for current sample
    
Invariant: combs.rows == hashCoefficients.columns (ordinary case only)
"""


