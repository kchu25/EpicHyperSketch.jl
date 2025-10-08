using EpicHyperSketch

println("=== DEBUGGING NEGATIVE HASH INDICES ===")

# Create a simple test case that might produce negative hash indices
activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()

# Create a sequence with large position gaps that might cause issues
seq_id = 1
# Large position differences to test hash calculation
motif_sequence = [(1, 10), (2, 100), (3, 1000)]  # Very large distances
sort!(motif_sequence, by=x->x[2])

activation_dict[seq_id] = [(filter=m, contribution=1.0, position=p) for (m, p) in motif_sequence]

println("Test sequence: $motif_sequence")

record = Record(activation_dict, 3; use_cuda=false, filter_len=8)

println("Hash coefficients shape: $(size(record.cms.hash_coeffs))")
println("Sketch shape: $(size(record.cms.sketch))")

# Let's manually test the hash calculation
include("../src/count_cpu.jl")

batch_idx = 1
refArray = record.vecRefArray[batch_idx]
hashCoefficients = record.cms.hash_coeffs
n = 1
sketch_row_ind = 1
comb_col_ind = 1
filter_len = 8

println("\nTesting hash calculation manually:")
println("RefArray for sequence $n:")
for i in 1:size(refArray, 1)
    filter_id = refArray[i, 1, n]
    position = refArray[i, 3, n]
    if filter_id != 0
        println("  Index $i: Filter $filter_id at position $position")
    end
end

# Test the calculate_conv_hash_cpu function
hash_result = EpicHyperSketch.calculate_conv_hash_cpu(
    record.combs, refArray, hashCoefficients, 
    comb_col_ind, sketch_row_ind, n, 
    size(record.combs, 1), filter_len
)

println("\nHash result: $hash_result")

if hash_result < 0
    println("❌ NEGATIVE HASH RESULT!")
else
    println("Hash result is positive: $hash_result")
    
    # Test the sketch index calculation
    num_counters = size(record.cms.sketch, 1) * size(record.cms.sketch, 2)
    num_cols_sketch = size(record.cms.sketch, 2)
    
    println("num_counters = $num_counters")
    println("num_cols_sketch = $num_cols_sketch")
    
    sketch_col_index_mod = hash_result % num_counters
    println("sketch_col_index_mod = $sketch_col_index_mod")
    
    final_index = (sketch_col_index_mod % num_cols_sketch) + 1
    println("final_index = $final_index")
    
    if final_index < 1 || final_index > num_cols_sketch
        println("❌ FINAL INDEX OUT OF BOUNDS!")
    else
        println("✓ Final index is valid")
    end
end

# Test with smaller positions to see if that fixes it
println("\n--- Testing with smaller positions ---")
activation_dict_small = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()
motif_sequence_small = [(1, 10), (2, 20), (3, 30)]  # Reasonable distances
activation_dict_small[seq_id] = [(filter=m, contribution=1.0, position=p) for (m, p) in motif_sequence_small]

record_small = Record(activation_dict_small, 3; use_cuda=false, filter_len=8)
refArray_small = record_small.vecRefArray[1]

hash_result_small = EpicHyperSketch.calculate_conv_hash_cpu(
    record_small.combs, refArray_small, record_small.cms.hash_coeffs, 
    1, 1, 1, 
    size(record_small.combs, 1), filter_len
)

println("Hash result with small positions: $hash_result_small")
