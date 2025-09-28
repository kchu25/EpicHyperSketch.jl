# Constants for refArray indexing (filters/features data)
const FILTER_INDEX_COLUMN = 1  # Filter ID / Feature ID
const POSITION_COLUMN = 2      # Position in sequence

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
    comb_col_ind = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sketch_row_ind = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    num_rows_combs, num_cols_combs = size(combs)
    num_rows_sketch, num_cols_sketch = size(sketch)
    num_counters = num_rows_sketch * num_cols_sketch

    if comb_col_ind ≤ num_cols_combs && sketch_row_ind ≤ num_rows_sketch && n ≤ size(refArray, 3)
        if is_combination_valid(combs, refArray, comb_col_ind, n)
            sketch_col_index = calculate_conv_hash(combs, refArray, hashCoefficients, 
                                                 comb_col_ind, sketch_row_ind, n, 
                                                 num_rows_combs, filter_len)
            if sketch_col_index != -1
                final_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
                CUDA.@atomic sketch[sketch_row_ind, final_index] += 1
            end
        end
    end
    return nothing
end

"""
CUDA kernel for ordinary counting of filter/feature combinations without position constraints.
"""
function count_kernel_ordinary(combs, refArray, hashCoefficients, sketch)
    comb_col_ind = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sketch_row_ind = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    num_rows_combs, num_cols_combs = size(combs)
    num_rows_sketch, num_cols_sketch = size(sketch)
    num_counters = num_rows_sketch * num_cols_sketch

    if comb_col_ind ≤ num_cols_combs && sketch_row_ind ≤ num_rows_sketch && n ≤ size(refArray, 3)
        if is_combination_valid(combs, refArray, comb_col_ind, n)
            sketch_col_index = calculate_ordinary_hash(combs, refArray, hashCoefficients, 
                                                     comb_col_ind, sketch_row_ind, n, 
                                                     num_rows_combs)
            final_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
            CUDA.@atomic sketch[sketch_row_ind, final_index] += 1
        end
    end
    return nothing
end

"""
Data Structure Specifications:

hashCoefficients: (num_hash_functions x "motif_size") matrix
    - Rows: number of hash functions (equals sketch height)
    - Columns: motif_size (ordinary) or 2 x motif_size - 1 (convolution)

combs: (motif_elements x num_combinations) matrix  
    - Rows: motif_size (ordinary) or ⌈motif_size/2⌉ (convolution)
    - Columns: total number of filter/feature combinations

Parameters:
    - motif_size: number of filters/features per motif
    - n: batch index for current sample
    
Invariant: combs.rows == hashCoefficients.columns (ordinary case only)
"""