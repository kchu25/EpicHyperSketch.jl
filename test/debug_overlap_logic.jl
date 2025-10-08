using EpicHyperSketch

println("=== INVESTIGATING POSITION-BASED OVERLAP LOGIC ===")

# Create test data with only the motif [22, 8, 39]
activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()

# Create just ONE sequence for detailed analysis
seq_id = 1
motif_sequence = [(22, 8), (8, 20), (39, 35)]
sort!(motif_sequence, by=x->x[2])

# Convert to ConvolutionFeature vector  
activation_dict[seq_id] = [(filter=m, contribution=1.0, position=p) for (m, p) in motif_sequence]

println("Test sequence: $motif_sequence")
println("After sorting by position: $motif_sequence")

record = Record(activation_dict, 3; use_cuda=false, filter_len=8)

batch_idx = 1
refArray = record.vecRefArray[batch_idx]
n = 1

println("\n=== POSITION AND OVERLAP ANALYSIS ===")
filter_len = 8

# Check positions and distances for combination [1, 2, 3]
comb_indices = [1, 2, 3]
println("Checking combination [1, 2, 3]:")
for (i, idx) in enumerate(comb_indices)
    filter_id = refArray[idx, 1, n]
    position = refArray[idx, 3, n]
    println("  Index $idx: Filter $filter_id at position $position")
end

# Calculate distances manually
println("\nDistance calculations (filter_len = $filter_len):")
for i in 1:(length(comb_indices)-1)
    idx1 = comb_indices[i]
    idx2 = comb_indices[i+1]
    
    filter1 = refArray[idx1, 1, n]
    filter2 = refArray[idx2, 1, n]
    pos1 = refArray[idx1, 3, n]
    pos2 = refArray[idx2, 3, n]
    
    distance = pos2 - pos1 - filter_len
    
    println("  Between index $idx1 (filter $filter1, pos $pos1) and index $idx2 (filter $filter2, pos $pos2):")
    println("    Distance = $pos2 - $pos1 - $filter_len = $distance")
    
    if distance < 0
        println("    ❌ OVERLAP DETECTED! This would cause the combination to be rejected.")
    else
        println("    ✓ No overlap, distance is valid.")
    end
end

# Test the actual hash calculation using the CPU function
println("\n=== TESTING HASH CALCULATION ===")

# We need to import the internal functions
include("../src/count_cpu.jl")

# Test the convolution hash calculation
sketch_row_ind = 1
comb_col_ind = 1  # First combination [1, 2, 3]
hashCoefficients = record.cms.hash_coeffs

println("Testing calculate_conv_hash_cpu for combination [1, 2, 3]:")
println("Sketch row: $sketch_row_ind")
println("Combination column: $comb_col_ind")
println("Hash coefficients shape: $(size(hashCoefficients))")

hash_result = EpicHyperSketch.calculate_conv_hash_cpu(
    record.combs, refArray, hashCoefficients, 
    comb_col_ind, sketch_row_ind, n, 
    size(record.combs, 1), filter_len
)

if hash_result == -1
    println("❌ HASH CALCULATION RETURNED -1 (INVALID/OVERLAP)")
    println("This means the combination is being rejected during counting!")
else
    println("✅ Hash calculation successful: $hash_result")
    println("This combination should be counted in the sketch.")
end

# Test is_combination_valid_cpu
println("\n=== TESTING COMBINATION VALIDITY ===")
is_valid = EpicHyperSketch.is_combination_valid_cpu(record.combs, refArray, comb_col_ind, n)
if is_valid
    println("✅ Combination [1, 2, 3] is valid for sequence $n")
else
    println("❌ Combination [1, 2, 3] is INVALID for sequence $n")
end

println("\n=== MANUAL STEP-BY-STEP HASH CALCULATION ===")
# Let's do the hash calculation step by step to see where it fails
sketch_col_index = Int32(0)

for elem_idx in 1:size(record.combs, 1)
    comb_index_value = record.combs[elem_idx, comb_col_ind]
    println("Element $elem_idx: comb_index_value = $comb_index_value")
    
    # Guard against invalid comb index
    if comb_index_value == 0 || comb_index_value > size(refArray, 1)
        println("  ❌ Invalid comb index!")
        break
    end
    
    filter_index = refArray[comb_index_value, 1, n]  # FILTER_INDEX_COLUMN = 1
    hash_coeff = hashCoefficients[sketch_row_ind, 2*(elem_idx-1)+1]
    sketch_col_index += filter_index * hash_coeff
    
    println("  filter_index = $filter_index")
    println("  hash_coeff = $hash_coeff")
    println("  sketch_col_index += $filter_index * $hash_coeff = $(filter_index * hash_coeff)")
    println("  running sketch_col_index = $sketch_col_index")
    
    if elem_idx < size(record.combs, 1)
        next_comb_index_value = record.combs[elem_idx+1, comb_col_ind]
        println("  next_comb_index_value = $next_comb_index_value")
        
        # Guard against invalid next index
        if next_comb_index_value == 0 || next_comb_index_value > size(refArray, 1)
            println("  ❌ Invalid next comb index!")
            break
        end
        
        position1 = refArray[comb_index_value, 3, n]  # POSITION_COLUMN = 3
        position2 = refArray[next_comb_index_value, 3, n]
        distance = position2 - position1 - filter_len
        
        println("  position1 = $position1, position2 = $position2")
        println("  distance = $position2 - $position1 - $filter_len = $distance")
        
        if distance < 0  # overlapping filters
            println("  ❌ OVERLAP! Returning -1")
            break
        else
            println("  ✅ No overlap")
        end
        
        distance_hash_coeff = hashCoefficients[sketch_row_ind, 2*elem_idx]
        sketch_col_index += distance_hash_coeff * distance
        
        println("  distance_hash_coeff = $distance_hash_coeff")
        println("  sketch_col_index += $distance_hash_coeff * $distance = $(distance_hash_coeff * distance)")
        println("  final sketch_col_index = $sketch_col_index")
    end
end
