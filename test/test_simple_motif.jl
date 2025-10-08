using EpicHyperSketch
using DataFrames

println("=== TESTING SIMPLE MOTIF [22, 8, 39] DETECTION ===")

# Create minimal test case with just one sequence containing the motif
activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()

# Insert motif [22,8,39] at positions [8, 20, 35]
motif_features = [(22, 8), (8, 20), (39, 35)]
conv_features = [(filter=f, contribution=1.0f0, position=p) for (f, p) in motif_features]
activation_dict[1] = conv_features

println("Created test data:")
println("  Sequence 1: ", [(feat.filter, feat.position) for feat in conv_features])

# Test with CPU counting
println("\n--- Testing CPU counting ---")
config = EpicHyperSketch.default_config(min_count=1, use_cuda=false)

result = EpicHyperSketch.obtain_enriched_configurations_cpu(
    activation_dict; 
    motif_size=3,
    filter_len=8,
    min_count=1,
    config=config
)

println("CPU Results: $(nrow(result)) motifs found")
if nrow(result) > 0
    for row in eachrow(result)
        motif = [row.m1, row.m2, row.m3]
        println("  Found: $motif (count: $(row.count))")
    end
else
    println("  No motifs found!")
end

# Now test with GPU if available
if EpicHyperSketch.CUDA.functional()
    println("\n--- Testing GPU counting ---")
    config_gpu = EpicHyperSketch.default_config(min_count=1, use_cuda=true)
    
    result_gpu = EpicHyperSketch.obtain_enriched_configurations(
        activation_dict; 
        motif_size=3,
        filter_len=8,
        config=config_gpu
    )
    
    println("GPU Results: $(nrow(result_gpu)) motifs found")
    if nrow(result_gpu) > 0
        for row in eachrow(result_gpu)
            motif = [row.m1, row.m2, row.m3]
            println("  Found: $motif (count: $(row.count))")
        end
    else
        println("  No motifs found!")
    end
else
    println("\n--- GPU not available ---")
end

# Let's also test with 2 occurrences to see if that changes anything
activation_dict[2] = conv_features  # Duplicate the same motif

println("\n=== Testing with 2 occurrences ===")

result2 = EpicHyperSketch.obtain_enriched_configurations_cpu(
    activation_dict; 
    motif_size=3,
    filter_len=8,
    min_count=1,
    config=config
)

println("CPU Results with 2 occurrences: $(nrow(result2)) motifs found")
for row in eachrow(result2)
    motif = [row.m1, row.m2, row.m3]
    println("  Found: $motif (count: $(row.count))")
end
