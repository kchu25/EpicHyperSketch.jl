using EpicHyperSketch
using Random
using DataFrames

# Debug the missing [22, 8, 39] motif
function debug_missing_motif()
    Random.seed!(42)
    
    # Create a simple test with just the problematic motif
    activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()
    filter_len = 8
    
    # Add the [22, 8, 39] motif 20 times to make it very obvious
    for i in 1:20
        motif_features = [(22, 8), (8, 20), (39, 35)]
        activation_dict[i] = [(filter=f, contribution=1.0f0, position=p) for (f, p) in motif_features]
    end
    
    println("=== DEBUG: Testing [22, 8, 39] motif ===")
    println("Created 20 instances of motif [22, 8, 39] at positions [8, 20, 35]")
    
    # Manual verification - count the motif
    count = 0
    for (seq_id, sequence) in activation_dict
        filter_ids = [feat.filter for feat in sequence]
        if Set(filter_ids) == Set([22, 8, 39])
            count += 1
            println("Found motif in sequence $seq_id: $filter_ids")
        end
    end
    println("Manual count: $count")
    
    # Test with EpicHyperSketch
    config = EpicHyperSketch.default_config(min_count=5, use_cuda=false)
    
    result = EpicHyperSketch.obtain_enriched_configurations_cpu(
        activation_dict;
        motif_size=3,
        filter_len=filter_len,
        min_count=5,
        config=config
    )
    
    println("EpicHyperSketch result:")
    println(result)
    
    if nrow(result) == 0
        println("❌ No motifs found by EpicHyperSketch!")
    else
        found_target = false
        for row in eachrow(result)
            motif = [row.m1, row.m2, row.m3]
            if Set(motif) == Set([22, 8, 39])
                found_target = true
                println("✓ Found target motif: $motif")
            end
        end
        if !found_target
            println("❌ Target motif [22, 8, 39] not found in results")
        end
    end
end

debug_missing_motif()
