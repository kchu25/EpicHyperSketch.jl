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
function count_kernel(combs, refArray, hashCoefficients, sketch, num_counters, num_cols_sketch, filter_len)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    # i: combination index
    # j: row index in the sketch
    # n: within_batch_index
    I = size(combs, 2)
    J = size(sketch, 1)
    N = size(refArray, 3)
    if i ≤ I && j ≤ J && n ≤ N
        valid = true
        # ensure all elements in the refArray according to the given combination are valid (non-zero)
        @inbounds for elem_idx in axes(combs, 1)
            if refArray[combs[elem_idx, i], 1, n] == 0
                valid = false
                break
            end
        end
        sketch_col_index::Int32 = 0
        if valid
            # Perform counting operation
            for elem_idx in axes(combs, 1)
                # get the filter number and times the hash coefficient
                comb_index_value = combs[elem_idx, i]
                filter_index = refArray[comb_index_value, 2, n]
                hash_coeff = hashCoefficients[j, 2*(elem_idx-1)+1]
                sketch_col_index += filter_index * hash_coeff

                if elem_idx < size(combs, 1)
                    # compute the distance between the occurrence of two filters
                    next_comb_index_value = combs[elem_idx+1, i]
                    position1 = refArray[comb_index_value,1,n]
                    position2 = refArray[next_comb_index_value,1,n]
                    # take into account of the length of each filter
                    _distance_ = position2 - position1 - filter_len
                    if _distance_ < 0
                        break
                    end
                    sketch_col_index += hashCoefficients[j, 2*elem_idx] * _distance_
                end
            end
            # get the column index; +1 to adjust to 1-base indexing
            sketch_col_index = ((sketch_col_index % num_counters) % num_cols_sketch) + 1
            # counter increment
            CUDA.@atomic sketch[j, sketch_col_index] += 1
        end
    end 
    return nothing
end


