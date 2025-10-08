using EpicHyperSketch
using DataFrames

println("=== ANALYZING MOTIF [22, 8, 39] DISTANCES ===")

# Let's manually check the distance calculation
motif_positions = [8, 20, 35]
filter_len = 8

println("Positions: $motif_positions")
println("Filter length: $filter_len")

dist1 = motif_positions[2] - motif_positions[1] - filter_len
dist2 = motif_positions[3] - motif_positions[2] - filter_len

println("Distance 1: pos[2] - pos[1] - filter_len = $motif_positions[2] - $motif_positions[1] - $filter_len = $dist1")
println("Distance 2: pos[3] - pos[2] - filter_len = $motif_positions[3] - $motif_positions[2] - $filter_len = $dist2")

if dist1 >= 0 && dist2 >= 0
    println("✓ All distances are non-negative, so no overlaps")
else
    println("✗ Negative distances detected - overlapping filters!")
end

# Let's also check what happens if we modify the positions slightly
println("\n=== Testing with different positions ===")

# Try with more spacing
test_cases = [
    ([10, 25, 40], "More spacing"),
    ([8, 20, 35], "Original positions"),
    ([5, 15, 25], "Closer spacing"),
    ([1, 10, 20], "Even closer spacing")
]

for (positions, desc) in test_cases
    println("\n$desc: $positions")
    d1 = positions[2] - positions[1] - filter_len
    d2 = positions[3] - positions[2] - filter_len
    println("  Distances: [$d1, $d2]")
    
    if d1 >= 0 && d2 >= 0
        println("  ✓ Valid (no overlaps)")
        
        # Test this configuration
        activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()
        conv_features = [(filter=22, contribution=1.0f0, position=positions[1]),
                         (filter=8, contribution=1.0f0, position=positions[2]),
                         (filter=39, contribution=1.0f0, position=positions[3])]
        activation_dict[1] = conv_features
        activation_dict[2] = conv_features  # Two occurrences
        
        config = EpicHyperSketch.default_config(min_count=1, use_cuda=false)
        result = EpicHyperSketch.obtain_enriched_configurations_cpu(
            activation_dict; 
            motif_size=3,
            filter_len=8,
            min_count=1,
            config=config
        )
        
        println("  → Found $(nrow(result)) motifs")
    else
        println("  ✗ Invalid (overlapping filters)")
    end
end
