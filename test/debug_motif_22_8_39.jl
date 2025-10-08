using EpicHyperSketch
using Random
using Combinatorics

# Helper function to convert filter IDs to ConvolutionFeature format with specific positions
function make_convolution_features_with_positions(filters_and_positions::Vector{Tuple{Int,Int}}, filter_len::Int=8)
    return [(filter=f, contribution=1.0f0, position=p) for (f, p) in filters_and_positions]
end

println("=== DEBUGGING MOTIF [22, 8, 39] ===")

# Create a minimal test case with just the problematic motif
activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()
filter_len = 8

# Insert motif [22,8,39] exactly as in the test, but with more detail
motif_features = [(22, 8), (8, 20), (39, 35)]
println("Original motif_features: ", motif_features)

# Sort by position as done in test
sort!(motif_features, by=x->x[2])
println("Sorted motif_features: ", motif_features)

# Create convolution features
conv_features = make_convolution_features_with_positions(motif_features, filter_len)
println("ConvolutionFeature objects:")
for (i, feat) in enumerate(conv_features)
    println("  [$i]: filter=$(feat.filter), position=$(feat.position)")
end

# Insert into activation dict
activation_dict[1] = conv_features

# Get max_active_len and create record
max_active_len = EpicHyperSketch.get_max_active_len(activation_dict)
println("max_active_len: ", max_active_len)

# Check how the refArray is constructed
vecRefArray, vecRefArrayContrib = EpicHyperSketch.constructVecRefArrays(
    activation_dict, max_active_len; 
    batch_size=500, case=:Convolution, use_cuda=false
)

println("\nrefArray content for sequence 1:")
ref_array = vecRefArray[1]
for i in 1:max_active_len
    filter_val = ref_array[i, 1, 1]  # FILTER_INDEX_COLUMN
    position_val = ref_array[i, 2, 1]  # POSITION_COLUMN
    if filter_val > 0
        println("  Index $i: filter=$filter_val, position=$position_val")
    end
end

# Generate combinations for motif_size = 3
motif_size = 3
combs = EpicHyperSketch.generate_combinations(motif_size, max_active_len, use_cuda=false)
println("\nGenerated combinations (first 10):")
for col in 1:min(10, size(combs, 2))
    comb = combs[:, col]
    println("  Combination $col: $comb")
    
    # Check what filters this combination would select
    filters_in_comb = []
    positions_in_comb = []
    valid = true
    for idx in comb
        if idx <= max_active_len && ref_array[idx, 1, 1] > 0
            push!(filters_in_comb, ref_array[idx, 1, 1])
            push!(positions_in_comb, ref_array[idx, 2, 1])
        else
            valid = false
            break
        end
    end
    if valid && length(filters_in_comb) == 3
        println("    -> Filters: $filters_in_comb, Positions: $positions_in_comb")
        if Set(filters_in_comb) == Set([22, 8, 39])
            println("    *** FOUND MOTIF [22, 8, 39] in combination $col! ***")
        end
    end
end

# Check if combination [1,2,3] would give us the motif
if size(combs, 2) >= 1
    target_comb = combs[:, 1]  # Should be [1,2,3] for first combination
    println("\nChecking combination [1,2,3]:")
    filters = [ref_array[target_comb[i], 1, 1] for i in 1:3]
    positions = [ref_array[target_comb[i], 2, 1] for i in 1:3]
    println("  Filters: $filters")
    println("  Positions: $positions")
    
    # Check distances
    if length(positions) >= 2
        dist1 = positions[2] - positions[1] - filter_len
        dist2 = positions[3] - positions[2] - filter_len
        println("  Distance 1: $dist1, Distance 2: $dist2")
        
        if dist1 >= 0 && dist2 >= 0
            println("  *** Valid combination with no overlaps ***")
        else
            println("  !!! Invalid combination due to overlaps !!!")
        end
    end
end
