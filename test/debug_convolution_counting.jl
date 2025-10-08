using EpicHyperSketch
using DataFrames

println("=== DEBUGGING CONVOLUTION COUNTING STEP BY STEP ===")

# Create simple test case
activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()
conv_features = [(filter=22, contribution=1.0f0, position=8),
                 (filter=8, contribution=1.0f0, position=20),
                 (filter=39, contribution=1.0f0, position=35)]
activation_dict[1] = conv_features

config = EpicHyperSketch.default_config(min_count=1, use_cuda=false)

# Manually construct the Record to trace what happens
println("1. Building Record...")
max_active_len = EpicHyperSketch.get_max_active_len(activation_dict)
println("   Max active length: $max_active_len")

vecRefArray, vecRefArrayContrib = EpicHyperSketch.constructVecRefArrays(
    activation_dict, max_active_len; 
    batch_size=500, case=:Convolution, use_cuda=false
)

combs = EpicHyperSketch.generate_combinations(3, max_active_len, use_cuda=false)
println("   Generated $(size(combs, 2)) combinations for motif size 3")
println("   First few combinations: ", combs[:, 1:min(5, size(combs, 2))])

# Check refArray contents
ref_array = vecRefArray[1]
println("   RefArray contents:")
for i in 1:max_active_len
    filter_val = ref_array[i, 1, 1]
    data_val = ref_array[i, 2, 1] 
    pos_val = ref_array[i, 3, 1]
    println("     [$i]: filter=$filter_val, data_pt=$data_val, position=$pos_val")
end

# Test combination validity
println("\n2. Testing combination validity...")
comb_col_ind = 1  # First combination [1,2,3]
n = 1  # First sequence

println("   Testing combination $(combs[:, comb_col_ind]) for sequence $n")

valid = true
for elem_idx in 1:3
    comb_index_value = combs[elem_idx, comb_col_ind]
    println("     Element $elem_idx: comb_index=$comb_index_value")
    
    if comb_index_value == 0
        println("       ✗ Invalid: comb_index is 0")
        valid = false
        break
    end
    
    if comb_index_value > size(ref_array, 1)
        println("       ✗ Invalid: comb_index $comb_index_value > refArray size $(size(ref_array, 1))")
        valid = false
        break
    end
    
    filter_val = ref_array[comb_index_value, 1, n]
    if filter_val == 0
        println("       ✗ Invalid: filter value is 0")
        valid = false
        break
    end
    
    println("       ✓ Valid: filter=$filter_val")
end

println("   Overall validity: $valid")

if valid
    println("\n3. Testing distance calculation...")
    filter_len = 8
    
    # Simulate calculate_conv_hash_cpu
    sketch_col_index = Int32(0)
    
    for elem_idx in 1:3
        comb_index_value = combs[elem_idx, comb_col_ind]
        filter_index = ref_array[comb_index_value, 1, n]
        
        println("     Element $elem_idx: comb_index=$comb_index_value, filter=$filter_index")
        
        if elem_idx < 3
            next_comb_index_value = combs[elem_idx+1, comb_col_ind]
            position1 = ref_array[comb_index_value, 3, n]
            position2 = ref_array[next_comb_index_value, 3, n]
            distance = position2 - position1 - filter_len
            
            println("       Distance calc: pos2=$position2 - pos1=$position1 - filter_len=$filter_len = $distance")
            
            if distance < 0
                println("       ✗ INVALID: Overlapping filters (distance=$distance)")
                valid = false
                break
            else
                println("       ✓ Valid distance: $distance")
            end
        end
    end
    
    println("   Final validity after distance check: $valid")
end
