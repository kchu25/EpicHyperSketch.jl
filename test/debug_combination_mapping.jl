using EpicHyperSketch
using DataFrames
using Random
using Combinatorics

println("=== INVESTIGATING COMBINATION GENERATION AND MAPPING ===")

# Create test data with only the motif [22, 8, 39]
Random.seed!(42)
activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()
sequences = []

# Create 12 sequences, each containing exactly the motif [22, 8, 39]
for seq_id in 1:12
    # Always insert the target motif at the same relative positions for consistency
    motif_sequence = [(22, 8), (8, 20), (39, 35)]
    
    # Add some noise (random additional features)
    num_additional = rand(0:2)
    for _ in 1:num_additional
        noise_filter = rand(400:500)  # Use high numbers to avoid conflicts
        noise_position = rand(40:100)
        push!(motif_sequence, (noise_filter, noise_position))
    end
    
    # Sort by position
    sort!(motif_sequence, by=x->x[2])
    
    # Convert to ConvolutionFeature vector  
    activation_dict[seq_id] = [(filter=m, contribution=1.0, position=p) for (m, p) in motif_sequence]
    push!(sequences, motif_sequence)
end

println("Created test data with motif [22, 8, 39] appearing exactly 12 times")

# Create the feature structure for Record construction
feature_data = (
    activation_dict=activation_dict, 
    motif_size=3,      # motif_size
    filter_len=8,      # filter_len
    min_count=1,       # min_count
    epsilon=1.0,       # epsilon
    delta=0.1          # delta
)

println("\nConstructing Record...")
record = Record(activation_dict, 3; use_cuda=false, filter_len=8)

println("\n=== DEBUGGING COMBINATION GENERATION ===")
println("Motif size: $(record.motif_size)")
println("Case: $(record.case)")
println("Number of combinations: $(size(record.combs, 2))")
println("Combination matrix shape: $(size(record.combs))")

# Print first few combinations
println("\nFirst 10 combinations (as indices):")
for i in 1:min(10, size(record.combs, 2))
    comb = [record.combs[j, i] for j in 1:size(record.combs, 1)]
    println("  Comb $i: $comb")
end

# Let's look at a specific batch and see how combinations map to actual filters
batch_idx = 1
refArray = record.vecRefArray[batch_idx]
println("\n=== REFARRAY ANALYSIS FOR BATCH $batch_idx ===")
println("RefArray shape: $(size(refArray))")

# Look at the first sequence in the batch
n = 1
println("\nSequence $n filters and positions:")
for i in 1:size(refArray, 1)
    filter_id = refArray[i, 1, n]  # FILTER_INDEX_COLUMN = 1
    position = refArray[i, 3, n]   # POSITION_COLUMN = 3
    if filter_id != 0
        println("  Index $i: Filter $filter_id at position $position")
    end
end

# Now check if combination [1, 2, 3] maps to filters [22, 8, 39]
println("\n=== CHECKING COMBINATION [1, 2, 3] MAPPING ===")
comb_indices = [1, 2, 3]
filters_in_comb = []
positions_in_comb = []
for idx in comb_indices
    filter_id = refArray[idx, 1, n]
    position = refArray[idx, 3, n]  
    push!(filters_in_comb, filter_id)
    push!(positions_in_comb, position)
    println("  Comb index $idx -> Filter $filter_id at position $position")
end
println("Combination [1, 2, 3] maps to filters $filters_in_comb at positions $positions_in_comb")

target_filters = Set([22, 8, 39])
mapped_filters = Set(filters_in_comb)
if mapped_filters == target_filters
    println("✓ Combination [1, 2, 3] correctly maps to target motif [22, 8, 39]")
else
    println("✗ Combination [1, 2, 3] maps to $mapped_filters, not target $target_filters")
end

# Check if there's any combination that maps to [22, 8, 39]
println("\n=== SEARCHING FOR COMBINATION THAT MAPS TO [22, 8, 39] ===")
found_combination = false
for comb_col in 1:size(record.combs, 2)
    comb_indices = [record.combs[j, comb_col] for j in 1:size(record.combs, 1)]
    
    # Check if this combination is valid for sequence n
    valid = true
    filters_in_comb = []
    for idx in comb_indices
        if idx > size(refArray, 1) || refArray[idx, 1, n] == 0
            valid = false
            break
        end
        push!(filters_in_comb, refArray[idx, 1, n])
    end
    
    if valid && Set(filters_in_comb) == target_filters
        println("✓ Found combination $comb_col: indices $comb_indices -> filters $filters_in_comb")
        found_combination = true
    end
end

if !found_combination
    println("✗ NO combination found that maps to filters [22, 8, 39] for sequence $n")
end

# Let's check all sequences in the first batch
println("\n=== CHECKING ALL SEQUENCES IN BATCH $batch_idx ===")
for seq_n in 1:size(refArray, 3)
    println("\nSequence $seq_n:")
    active_filters = []
    for i in 1:size(refArray, 1)
        filter_id = refArray[i, 1, seq_n]
        if filter_id != 0
            push!(active_filters, filter_id)
        end
    end
    println("  Active filters: $active_filters")
    
    # Check if [22, 8, 39] are all present
    if 22 in active_filters && 8 in active_filters && 39 in active_filters
        println("  ✓ Contains target motif [22, 8, 39]")
        
        # Find their indices in refArray
        indices_22_8_39 = []
        for i in 1:size(refArray, 1)
            filter_id = refArray[i, 1, seq_n]
            if filter_id in [22, 8, 39]
                push!(indices_22_8_39, i)
            end
        end
        println("  Target filters are at refArray indices: $indices_22_8_39")
        
        # Check if there's a combination with these indices
        target_set = Set(indices_22_8_39)
        found_comb = false
        for comb_col in 1:size(record.combs, 2)
            comb_indices = Set([record.combs[j, comb_col] for j in 1:size(record.combs, 1)])
            if comb_indices == target_set
                println("  ✓ Combination $comb_col matches: $(sort(collect(comb_indices)))")
                found_comb = true
                break
            end
        end
        if !found_comb
            println("  ✗ NO combination matches indices $indices_22_8_39")
        end
    else
        println("  ✗ Does not contain complete target motif [22, 8, 39]")
    end
end
