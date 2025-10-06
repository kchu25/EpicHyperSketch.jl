"""
    cms_rows(delta)
Calculate the number of rows in the Count-Min Sketch given the error probability `delta`.
"""
function cms_rows(delta::Float64)
    return Int(ceil(log(1 / delta)))
end

"""
    cms_num_counters(rows, epsilon)
Calculate the total number of counters in the sketch given `rows` and error tolerance `epsilon`.

Note: The counters simply means the total number of cells in the sketch matrix.
"""
function cms_num_counters(rows::Int, epsilon::Float64)
    return rows * Int(ceil(exp(1) / epsilon))
end

"""
    cms_cols(num_counters, rows)
Calculate the number of columns in the sketch given `num_counters` and `rows`.
"""
function cms_cols(num_counters::Int, rows::Int)
    return num_counters ÷ rows
end


"""
    CountMinSketch(motif_size::Integer; delta, epsilon, use_cuda, seed)

Create a CountMinSketch object.

- motif_size: Number of features (motifs).
- delta: Error probability.
- epsilon: Error tolerance.
- use_cuda: If true, use CUDA arrays.
- seed: Random seed for reproducibility (optional).
"""
mutable struct CountMinSketch
    hash_coeffs::AbstractMatrix{IntType}
    sketch::AbstractMatrix{IntType}

    function CountMinSketch(
        motif_size::Integer; 
        delta=DEFAULT_CMS_DELTA, 
        epsilon=DEFAULT_CMS_EPSILON, 
        case::Symbol=:OrdinaryFeatures,
        use_cuda::Bool=true,
        seed::Union{Int, Nothing}=nothing
    )
        @assert motif_size > 0 "motif_size must be a positive integer"
        @assert 0 < delta < 1 "delta must be in (0, 1)"
        @assert 0 < epsilon < 1 "epsilon must be in (0, 1)"
        @assert case in (:OrdinaryFeatures, :Convolution) "case must be :OrdinaryFeatures or :Convolution"

        rows = cms_rows(delta)
        num_counters = cms_num_counters(rows, epsilon)
        cols = cms_cols(num_counters, rows)
        sketch = zeros(IntType, rows, cols)

        num_hash_cols = num_hash_columns(motif_size, case)
        
        # Set seed if provided for reproducibility
        if seed !== nothing
            Random.seed!(seed)
        end
        
        rand_matrix = rand(1:num_counters-1, (rows, num_hash_cols))
        hash_coeffs = IntType.(rand_matrix)

        if use_cuda
            hash_coeffs = CuArray(hash_coeffs)
            sketch = CuArray(sketch)
        end

        return new(hash_coeffs, sketch)
    end
end

"""
    num_hash_columns(motif_size, case)
Return the number of hash columns for the given motif size and case.
"""
function num_hash_columns(motif_size::Integer, case::Symbol)
    case == :OrdinaryFeatures && return motif_size
    case == :Convolution && return 2 * motif_size - 1
    error("Unsupported case: $case")
end

import Base: show, getindex, size, iterate

"""
    size(cms::CountMinSketch)
Return the size of the sketch matrix.
"""
function size(cms::CountMinSketch)
    return size(cms.sketch)
end

"""
    show(io::IO, cms::CountMinSketch)
Custom pretty printing for CountMinSketch objects.
"""
function show(io::IO, cms::CountMinSketch)
    println(io, "CountMinSketch(")
    println(io, "  sketch: ", size(cms.sketch))
    print(io, ")")
end

"""
    getindex(cms::CountMinSketch, i, j)
Access the sketch matrix at position (i, j).
"""
function getindex(cms::CountMinSketch, i::Int, j::Int)
    return cms.sketch[i, j]
end
