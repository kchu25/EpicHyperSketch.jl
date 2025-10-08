using EpicHyperSketch
using DataFrames

# Simple test to debug convolution counting
function test_simple_convolution()
    println("=== SIMPLE CONVOLUTION DEBUG TEST ===")
    
    filter_len = 8
    motif_size = 2
    
    # Create a simple dictionary with known motifs
    # Motif: filters [5, 10] with positions that don't overlap
    # Position 1 = 10, Position 2 = 30
    # Distance = 30 - 10 - 8 = 12
    
    activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()
    
    # Add 20 sequences with the exact same motif
    for i in 1:20
        activation_dict[i] = [
            (filter=5, contribution=1.0f0, position=10),
            (filter=10, contribution=1.0f0, position=30)
        ]
    end
    
    # Add some noise sequences
    for i in 21:30
        activation_dict[i] = [
            (filter=rand(20:30), contribution=1.0f0, position=rand(1:50))
        ]
    end
    
    println("Created test dictionary with $(length(activation_dict)) sequences")
    println("Expected: filters [5,10] with distance = 30-10-8 = 12 should appear ~20 times")
    
    # Test with CPU
    config = EpicHyperSketch.default_config(min_count=10, use_cuda=false)
    
    result = EpicHyperSketch.obtain_enriched_configurations_cpu(
        activation_dict;
        motif_size=motif_size,
        filter_len=filter_len,
        min_count=10,
        config=config
    )
    
    println("\nResults:")
    println(result)
    
    if nrow(result) > 0
        println("\nFound $(nrow(result)) enriched motifs")
        for row in eachrow(result)
            println("  Filters: [$(row.m1), $(row.m2)], Distance: $(row.d12)")
        end
    else
        println("No enriched motifs found!")
    end
end

test_simple_convolution()
