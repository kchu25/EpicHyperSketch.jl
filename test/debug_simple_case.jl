using EpicHyperSketch
using DataFrames

println("=== INVESTIGATING WHY MOTIF [22, 8, 39] IS MISSED ===")

# Create the same test data but with verbose tracing
activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()

# Insert the problematic motif exactly 12 times
for i in 1:12
    motif_features = [(22, 8), (8, 20), (39, 35)]
    # Add different noise to each sequence to make them unique
    noise_n = rand(0:2)
    noise_features = [(rand(400:450) + i, rand(50:100)) for _ in 1:noise_n]
    all_features = [motif_features..., noise_features...]
    sort!(all_features, by=x->x[2])
    activation_dict[i] = [(filter=f, contribution=1.0f0, position=p) for (f, p) in all_features]
end

println("Created test data with motif [22, 8, 39] appearing exactly 12 times")
println("Sample sequences:")
for i in 1:3
    seq = activation_dict[i]
    filters_and_pos = [(feat.filter, feat.position) for feat in seq]
    println("  Seq $i: $filters_and_pos")
end

# Test with min_count = 1 (should definitely find it)
println("\nTesting with min_count = 1...")
config = EpicHyperSketch.default_config(min_count=1, use_cuda=false)

result = EpicHyperSketch.obtain_enriched_configurations_cpu(
    activation_dict; 
    motif_size=3,
    filter_len=8,
    min_count=1,
    config=config
)

println("Results: $(nrow(result)) motifs found")
if nrow(result) > 0
    println("Columns in result: $(names(result))")
    println("First few rows:")
    println(first(result, min(5, nrow(result))))
    
    global found_target = false
    for row in eachrow(result)
        # Check what columns are available
        if :m1 in names(result) && :m2 in names(result) && :m3 in names(result)
            motif = [row.m1, row.m2, row.m3]
            if hasproperty(row, :count)
                count_val = row.count
                println("  Found: $motif (count: $count_val)")
            else
                println("  Found: $motif (count: unknown)")
            end
            if Set(motif) == Set([22, 8, 39])
                global found_target = true
                println("  *** FOUND TARGET MOTIF! ***")
            end
        else
            println("  Row: $row")
        end
    end
    if !found_target
        println("  !!! TARGET MOTIF [22, 8, 39] NOT FOUND !!!")
    end
else
    println("  No motifs found at all!")
end

# Manual verification
println("\nManual verification of ground truth:")
global motif_count = 0
for (seq_id, sequence) in activation_dict
    filter_ids = [feat.filter for feat in sequence]
    if 22 in filter_ids && 8 in filter_ids && 39 in filter_ids
        global motif_count += 1
    end
end
println("Manual count of sequences containing [22, 8, 39]: $motif_count")

println("\nLet me check if motif [22, 8, 39] exists in results (different order):")
global found_any_22_8_39 = false
for row in eachrow(result)
    motif = Set([row.m1, row.m2, row.m3])
    if motif == Set([22, 8, 39])
        println("  Found [22, 8, 39] as [$(row.m1), $(row.m2), $(row.m3)] with contribution $(row.contribution)")
        global found_any_22_8_39 = true
    end
end
if !found_any_22_8_39
    println("  NO, motif [22, 8, 39] not found in ANY order in the results!")
end
