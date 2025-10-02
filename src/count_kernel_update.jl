

"""
Check if all filters/features in a combination are present (non-zero) in refArray.
"""
function is_combination_valid(combs, refArray, comb_col_ind, n)
    @inbounds for elem_idx in axes(combs, 1)
        if refArray[combs[elem_idx, comb_col_ind], FILTER_INDEX_COLUMN, n] == 0
            return false
        end
    end
    return true
end

"""
Calculate hash index for ordinary (non-convolution) counting method.
"""
function calculate_ordinary_hash(combs, refArray, hashCoefficients, comb_col_ind, sketch_row_ind, n, num_rows_combs)
    sketch_col_index = Int32(0)
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
    sketch_col_index = Int32(0)
    @inbounds for elem_idx in 1:num_rows_combs
        comb_index_value = combs[elem_idx, comb_col_ind]
        filter_index = refArray[comb_index_value, FILTER_INDEX_COLUMN, n]
        hash_coeff = hashCoefficients[sketch_row_ind, 2*(elem_idx-1)+1]
        sketch_col_index += filter_index * hash_coeff

        if elem_idx < num_rows_combs
            next_comb_index_value = combs[elem_idx+1, comb_col_ind]
            position1 = refArray[comb_index_value, POSITION_COLUMN, n]
            position2 = refArray[next_comb_index_value, POSITION_COLUMN, n]
            distance = position2 - position1 - filter_len
            
            if distance < 0  # overlapping filters
                return -1  # signal invalid
            end
            
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
    
    num_rows_combs, num_cols_combs = size(combs)
    num_rows_sketch, num_cols_sketch = size(sketch)
    num_counters = num_rows_sketch * num_cols_sketch
    
    in_bounds = (comb_col_ind ≤ num_cols_combs && 
                 sketch_row_ind ≤ num_rows_sketch && 
                 n ≤ size(refArray, 3))
    
    return (comb_col_ind, sketch_row_ind, n, num_counters, num_cols_sketch, in_bounds)
end

# Common sketch index calculation and update logic
function _process_sketch_update!(sketch, sketch_row_ind, sketch_col_index, num_counters, num_cols_sketch)
    final_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
    CUDA.@atomic sketch[sketch_row_ind, final_index] += 1
end

# Common sketch index calculation and candidate selection logic
function _process_candidate_selection!(sketch, selectedCombs, sketch_row_ind, sketch_col_index, 
                                     num_counters, num_cols_sketch, comb_col_ind, n, min_count)
    final_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
    if sketch[sketch_row_ind, final_index] ≥ min_count
        selectedCombs[comb_col_ind, n] = true
    end
end

"""
CUDA kernel for convolution-based candidate selection with position-aware hashing.
Identifies combinations that meet minimum count threshold and marks them in selectedCombs.
"""
function count_kernel_conv_get_candidates(combs, refArray, hashCoefficients, sketch, filter_len, selectedCombs, min_count)
    comb_col_ind, sketch_row_ind, n, num_counters, num_cols_sketch, in_bounds = 
        _kernel_setup_and_bounds_check(combs, refArray, sketch)
    
    if in_bounds && is_combination_valid(combs, refArray, comb_col_ind, n)
        sketch_col_index = calculate_conv_hash(combs, refArray, hashCoefficients, 
                                             comb_col_ind, sketch_row_ind, n, 
                                             size(combs, 1), filter_len)
        if sketch_col_index != -1
            _process_candidate_selection!(sketch, selectedCombs, sketch_row_ind, sketch_col_index,
                                        num_counters, num_cols_sketch, comb_col_ind, n, min_count)
        end
    end
    return nothing
end

"""
CUDA kernel for ordinary candidate selection without position constraints.
Identifies combinations that meet minimum count threshold and marks them in selectedCombs.
"""
function count_kernel_ordinary_get_candidate(combs, refArray, hashCoefficients, sketch, selectedCombs, min_count)
    comb_col_ind, sketch_row_ind, n, num_counters, num_cols_sketch, in_bounds = 
        _kernel_setup_and_bounds_check(combs, refArray, sketch)
    
    if in_bounds && is_combination_valid(combs, refArray, comb_col_ind, n)
        sketch_col_index = calculate_ordinary_hash(combs, refArray, hashCoefficients, 
                                                 comb_col_ind, sketch_row_ind, n, 
                                                 size(combs, 1))
        _process_candidate_selection!(sketch, selectedCombs, sketch_row_ind, sketch_col_index,
                                    num_counters, num_cols_sketch, comb_col_ind, n, min_count)
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
function obtain_motifs_conv!(CindsVec, combs, refArray, motifs_obtained, filter_len)
    i, j, n, in_bounds = _config_kernel_setup(CindsVec)
    
    if in_bounds
        K = size(combs, 1)
        @inbounds for k = 1:K
            # Store filter ID
            motifs_obtained[i, 2*(k-1)+1] = refArray[combs[k, j], FILTER_INDEX_COLUMN, n]

            # Store distance to next filter (if not last)
            if k < K
                pos1 = refArray[combs[k, j], POSITION_COLUMN, n]
                pos2 = refArray[combs[k+1, j], POSITION_COLUMN, n]
                motifs_obtained[i, 2*k] = pos2 - pos1 - filter_len
            end
        end
    end
    return nothing
end

"""
Extract configurations for ordinary case: only filter/feature IDs.
"""
function obtain_motifs_ordinary!(CindsVec, combs, refArray, motifs_obtained)
    i, j, n, in_bounds = _config_kernel_setup(CindsVec)
    
    if in_bounds
        K = size(combs, 1)
        @inbounds for k = 1:K
            motifs_obtained[i, k] = refArray[combs[k, j], FILTER_INDEX_COLUMN, n]
        end
    end
    return nothing
end





"""
Data Structure Specifications:

hashCoefficients: (num_hash_functions x "motif_size") matrix
    - Rows: number of hash functions (equals sketch height)
    - Columns: motif_size (ordinary) or 2 x motif_size - 1 (convolution)

combs: (num_motif_elements x num_combinations) matrix  
    - Rows: motif_size (ordinary) or ⌈motif_size/2⌉ (convolution)
    - Columns: total number of filter/feature combinations

Parameters:
    - motif_size: number of filters/features per motif
    - n: batch index for current sample
    
Invariant: combs.rows == hashCoefficients.columns (ordinary case only)
"""