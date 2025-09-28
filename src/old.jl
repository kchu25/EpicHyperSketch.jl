"""
Note that combs is a matrix that contains combinations (horizontally), constructed 
e.g. by 
```
        reduce(hcat, combinations(1:5, 3))        
```
which gives
```
 1  1  1  1  1  1  2  2  2  3
 2  2  2  3  3  4  3  3  4  4
 3  4  5  4  5  5  4  5  5  5
```
We use the following indexing convention:
elem_idx is 1:3
and when fixing a column c, 
comb_index_value is the elem in the c-th column, i.e. combs[elem_idx, c]

"""

"""
    count_kernel(combs, refArray, hashCoefficients, sketch, num_counters, num_cols_sketch, filter_len)

A CUDA kernel function for counting combinations of filter occurrences in genomic sequences using 
count-min sketch data structure for approximate frequency estimation.

# Arguments
- `combs::CuMatrix`: Matrix of combinations where each column represents a combination of indices to count
- `refArray::CuArray{T,3}`: 3D reference array where:
  - `refArray[filter_idx, 1, batch_idx]` contains the position of filter occurrence (0 if invalid)
  - `refArray[filter_idx, 2, batch_idx]` contains the filter number/identifier
- `hashCoefficients::CuMatrix`: Hash coefficients matrix for sketch computation where:
  - Odd columns (2*elem_idx-1) contain filter hash coefficients
  - Even columns (2*elem_idx) contain distance hash coefficients
- `sketch::CuMatrix`: Count-min sketch matrix to store frequency counts (modified in-place)
- `num_counters::Integer`: Total number of available counters for hashing
- `num_cols_sketch::Integer`: Number of columns in the sketch matrix
- `filter_len::Integer`: Length of each filter (used for distance calculations)

# CUDA Thread Organization
- `i` (x-dimension): Combination index from `combs`
- `j` (y-dimension): Row index in the sketch matrix (hash function index)  
- `n` (z-dimension): Within-batch index in `refArray`

# Algorithm
1. **Validation**: Checks if all filters in the combination have valid positions (non-zero in refArray)
2. **Hash Computation**: For each valid combination:
   - Computes hash based on filter numbers and their hash coefficients
   - Adds distance-based hash terms for consecutive filter pairs
   - Distance = position_difference - filter_length (must be non-negative)
3. **Sketch Update**: Atomically increments the appropriate sketch counter

# Notes
- Uses atomic operations for thread-safe sketch updates
- Implements early termination if filters overlap (negative distance)
- Hash result is mapped to sketch columns using modular arithmetic
- Assumes 1-based indexing for Julia arrays

# Returns
`nothing` (modifies `sketch` in-place)

# Example Usage
```julia
# Launch kernel with appropriate grid/block dimensions
@cuda threads=(32, 8, 4) blocks=(ceil(Int, I/32), ceil(Int, J/8), ceil(Int, N/4)) \\
      count_kernel(combinations, ref_positions, hash_coeffs, sketch_matrix, 
                  n_counters, n_cols, filter_length)
```
"""
function count_kernel_conv(combs, refArray, hashCoefficients, sketch, filter_len)
    comb_col_ind = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    sketch_row_ind = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    # i: combination index
    # j: row index in the sketch
    # n: within_batch_index
    num_rows_combs, num_cols_combs = size(combs)
    num_rows_sketch, num_cols_sketch = size(sketch)
    num_counters = num_rows_sketch * num_cols_sketch

    N = size(refArray, 3)
    if comb_col_ind ≤ num_cols_combs && sketch_row_ind ≤ num_rows_sketch && n ≤ N
        valid = true
        # ensure all elements in the refArray according to the given combination are valid (non-zero)
        @inbounds for elem_idx in axes(combs, 1)
            if refArray[combs[elem_idx, comb_col_ind], 1, n] == 0
                valid = false
                break
            end
        end
        # determine which column in the sketch to increment
        sketch_col_index::Int32 = 0
        @inbounds if valid
            # Perform counting operation
            for elem_idx in 1:num_rows_combs
                # get the filter number and times the hash coefficient
                comb_index_value = combs[elem_idx, comb_col_ind]
                filter_index = refArray[comb_index_value, 1, n]
                hash_coeff = hashCoefficients[sketch_row_ind, 2*(elem_idx-1)+1]
                sketch_col_index += filter_index * hash_coeff

                if elem_idx < num_rows_combs
                    # compute the distance between the occurrence of two filters
                    next_comb_index_value = combs[elem_idx+1, comb_col_ind]
                    position1 = refArray[comb_index_value, 2, n]
                    position2 = refArray[next_comb_index_value, 2, n]
                    # take into account of the length of each filter
                    _distance_ = position2 - position1 - filter_len
                    if _distance_ < 0 # overlapping filters, skip this combination
                        break
                    end
                    sketch_col_index += hashCoefficients[sketch_row_ind, 2*elem_idx] * _distance_
                end
            end
            # get the column index; +1 to adjust to 1-base indexing
            sketch_col_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
            # counter increment
            CUDA.@atomic sketch[sketch_row_ind, sketch_col_index] += 1
        end
    end 
    return nothing
end


function count_kernel_ordinary(combs, refArray, hashCoefficients, sketch)
    comb_col_ind = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    sketch_row_ind = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    # i: combination index
    # j: row index in the sketch
    # n: within_batch_index
    num_rows_combs, num_cols_combs = size(combs)
    num_rows_sketch, num_cols_sketch = size(sketch)
    num_counters = num_rows_sketch * num_cols_sketch

    N = size(refArray, 3)
    if comb_col_ind ≤ num_cols_combs && sketch_row_ind ≤ num_rows_sketch && n ≤ N
        valid = true
        # ensure all elements in the refArray according to the given combination are valid (non-zero)
        @inbounds for elem_idx in axes(combs, 1)
            if refArray[combs[elem_idx, comb_col_ind], 1, n] == 0
                valid = false
                break
            end
        end
        # determine which column in the sketch to increment
        sketch_col_index::Int32 = 0
        @inbounds if valid
            # Perform counting operation
            for elem_idx in 1:num_rows_combs
                # get the filter number and times the hash coefficient
                comb_index_value = combs[elem_idx, comb_col_ind]
                filter_index = refArray[comb_index_value, 1, n]
                hash_coeff = hashCoefficients[sketch_row_ind, elem_idx]
                sketch_col_index += filter_index * hash_coeff
            end
            # get the column index; +1 to adjust to 1-base indexing
            sketch_col_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
            # counter increment
            CUDA.@atomic sketch[sketch_row_ind, sketch_col_index] += 1
        end
    end 
    return nothing
end


"""
Note:

    hashCoefficients is a matrix where 
        number of rows = number of hash functions = number of rows in the sketch
        number of columns is the "actual motifs_size" 

    comb is a matrix where 
        number of rows = 
            motif_size in the ordinary case
            (motifs_size+1) ÷ 2 in the convolutional case
        number of columns = number of combinations

    "motif_size" (num rows of combs) is equal the number of columns in hashCoefficients when not    using convolutional motifs

    n is the within batch index
"""




function count_kernel_shared(combs, refArray, hashCoefficients, sketch, filter_len=nothing)
    comb_col_ind = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sketch_row_ind = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    num_rows_combs, num_cols_combs = size(combs)
    num_rows_sketch, num_cols_sketch = size(sketch)
    num_counters = num_rows_sketch * num_cols_sketch
    N = size(refArray, 3)

    if comb_col_ind ≤ num_cols_combs && sketch_row_ind ≤ num_rows_sketch && n ≤ N
        # Check if combination is valid
        valid = true
        @inbounds for elem_idx in axes(combs, 1)
            if refArray[combs[elem_idx, comb_col_ind], 1, n] == 0
                valid = false
                break
            end
        end

        if valid
            sketch_col_index = Int32(0)
            is_conv = filter_len !== nothing
            
            @inbounds for elem_idx in 1:num_rows_combs
                comb_index_value = combs[elem_idx, comb_col_ind]
                filter_index = refArray[comb_index_value, 1, n]
                
                # Different hash coefficient indexing for conv vs ordinary
                hash_coeff = is_conv ? 
                    hashCoefficients[sketch_row_ind, 2*(elem_idx-1)+1] :
                    hashCoefficients[sketch_row_ind, elem_idx]
                    
                sketch_col_index += filter_index * hash_coeff

                # Distance calculation only for conv case
                if is_conv && elem_idx < num_rows_combs
                    next_comb_index_value = combs[elem_idx+1, comb_col_ind]
                    position1 = refArray[comb_index_value, 2, n]
                    position2 = refArray[next_comb_index_value, 2, n]
                    distance = position2 - position1 - filter_len
                    
                    if distance < 0  # overlapping filters
                        valid = false
                        break
                    end
                    
                    sketch_col_index += hashCoefficients[sketch_row_ind, 2*elem_idx] * distance
                end
            end
            
            if valid
                sketch_col_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
                CUDA.@atomic sketch[sketch_row_ind, sketch_col_index] += 1
            end
        end
    end
    return nothing
end

# Keep original function names for compatibility
count_kernel_conv(combs, refArray, hashCoefficients, sketch, filter_len) = 
    count_kernel_shared(combs, refArray, hashCoefficients, sketch, filter_len)

count_kernel_ordinary(combs, refArray, hashCoefficients, sketch) = 
    count_kernel_shared(combs, refArray, hashCoefficients, sketch)