using EpicHyperSketch
using Random
using Combinatorics  # For combinations function
using CUDA  # For CUDA.functional() check
using DataFrames  # For DataFrame output

# Helper function to convert integers to OrdinaryFeature format
function make_features(feature_ids::Vector{Int})
    return [(feature=id, contribution=1.0f0) for id in feature_ids]
end

"""
Create a large test ActivationDict with known ground truth motifs.
This example has 750 sequences (> BATCH_SIZE = 500) with controlled motif patterns.

Ground Truth for motif_size = 3:
- Motif [7, 19, 42]: appears exactly 25 times
- Motif [13, 28, 55]: appears exactly 15 times  
- Motif [3, 41, 67]: appears exactly 8 times
- Motif [22, 8, 39]: appears exactly 12 times
- All other 3-element combinations should be rare (≤ 2 occurrences)
"""
function create_large_test_dict()
    Random.seed!(42)  # For reproducibility
    activation_dict = Dict{Int, Vector{EpicHyperSketch.OrdinaryFeature}}()
    
    # Insert motif [7,19,42] exactly 25 times - mix consecutive and scattered
    for i in 1:25
        if i <= 8
            # Consecutive at beginning
            activation_dict[i] = make_features([7, 19, 42, rand(100:150, rand(2:5))...])
        elseif i <= 16
            # Scattered in sequence (elements present but not consecutive)
            filler1 = rand(100:150, rand(1:2))
            filler2 = rand(100:150, rand(1:2)) 
            filler3 = rand(100:150, rand(1:3))
            activation_dict[i] = make_features([filler1..., 7, filler2..., 19, filler3..., 42, rand(100:150, rand(1:2))...])
        elseif i <= 20
            # In middle (consecutive)
            activation_dict[i] = make_features([rand(100:150, rand(1:3))..., 7, 19, 42, rand(100:150, rand(1:3))...])
        else
            # Mixed: some consecutive, some scattered
            if rand() < 0.5
                activation_dict[i] = make_features([rand(100:150, rand(2:5))..., 7, 19, 42])
            else
                # Scattered version
                activation_dict[i] = make_features([7, rand(100:150, 2)..., 19, rand(100:150, 1)..., 42])
            end
        end
    end
    
    # Insert motif [13,28,55] exactly 15 times - similar mix
    for i in 26:40
        if i <= 31
            # Consecutive
            activation_dict[i] = make_features([rand(200:250, rand(1:3))..., 13, 28, 55, rand(200:250, rand(1:4))...])
        else
            # Scattered
            filler = rand(200:250, rand(1:3))
            activation_dict[i] = make_features([13, filler..., 28, rand(200:250, rand(1:2))..., 55, rand(200:250, rand(1:2))...])
        end
    end
    
    # Insert motif [3,41,67] exactly 8 times
    for i in 41:48
        if i <= 44
            activation_dict[i] = make_features([3, 41, 67, rand(300:350, rand(2:6))...])
        else
            # Scattered
            activation_dict[i] = make_features([3, rand(300:350, rand(2:4))..., 41, rand(300:350, 1)..., 67])
        end
    end
    
    # Insert motif [22,8,39] exactly 12 times
    for i in 49:60
        if i <= 54
            activation_dict[i] = make_features([rand(400:450, rand(1:2))..., 22, 8, 39, rand(400:450, rand(1:3))...])
        else
            # Scattered
            activation_dict[i] = make_features([22, rand(400:450, rand(1:3))..., 8, rand(400:450, rand(1:2))..., 39, rand(400:450, rand(1:2))...])
        end
    end
    
    # Fill sequences 61-600 with random data (no systematic motifs)
    for i in 61:600
        # Random sequences with length 3-8, using values 500-600 to avoid accidental motifs
        seq_length = rand(3:8)
        activation_dict[i] = make_features(rand(500:600, seq_length))
    end
    
    # Add sequences 601-700 with some overlapping elements but no complete motifs
    for i in 601:700
        # These might have 1 or 2 elements from our motifs, but not complete triplets
        base_elements = rand([7, 19, 13, 28, 3, 41, 22, 8], rand(1:2))
        filler = rand(700:800, rand(3:6))
        activation_dict[i] = make_features([base_elements..., filler...])
    end
    
    # Add sequences 701-750 with some empty and short sequences
    for i in 701:750
        if i <= 710
            activation_dict[i] = EpicHyperSketch.OrdinaryFeature[]  # Empty sequences
        elseif i <= 720
            activation_dict[i] = make_features([rand(900:1000)])  # Single elements
        elseif i <= 730
            activation_dict[i] = make_features([rand(900:1000), rand(900:1000)])  # Pairs
        else
            activation_dict[i] = make_features(rand(900:1000, rand(3:5)))  # Regular random sequences
        end
    end
    
    return activation_dict
end
"""
Verify ground truth by manually counting motifs in the test dictionary.
Supports both consecutive subsequences and subset-based motifs.
"""
function verify_ground_truth(activation_dict, motif_size=3; check_subsets=true)
    subset_motif_counts = Dict{Set{Int},Int}()

    for (seq_id, sequence) in activation_dict
        if length(sequence) >= motif_size
            # Extract feature IDs from NamedTuples
            feature_ids = [feat.feature for feat in sequence]

            # Count subset-based motifs (all combinations of motif_size elements)
            if check_subsets
                unique_seq = unique(feature_ids)
                if length(unique_seq) >= motif_size
                    for combo in combinations(unique_seq, motif_size)
                        motif_set = Set(combo)
                        subset_motif_counts[motif_set] = get(subset_motif_counts, motif_set, 0) + 1
                    end
                end
            end
        end
    end

    # Sort consecutive motifs by count (descending)


    if check_subsets
        # Sort subset motifs by count (descending)
        sorted_subsets = sort(collect(subset_motif_counts), by=x -> x[2], rev=true)

        println("\n=== GROUND TRUTH VERIFICATION (SUBSETS) ===")
        println("Top 15 subset motifs by count:")
        for (i, (motif_set, count)) in enumerate(sorted_subsets[1:min(15, end)])
            println("$i. $(collect(motif_set)): $count times")
        end
    end

    # Verify our expected motifs (consecutive)
    expected_consecutive_motifs = [
        ([7, 19, 42], 25),
        ([13, 28, 55], 15),
        ([22, 8, 39], 12),
        ([3, 41, 67], 8)
    ]

    if check_subsets
        println("\n=== EXPECTED SUBSET MOTIFS VERIFICATION ===")
        for (expected_motif, expected_count) in expected_consecutive_motifs
            motif_set = Set(expected_motif)
            actual_count = get(subset_motif_counts, motif_set, 0)
            status = actual_count == expected_count ? "✓" : "✗"
            println("$status $(collect(motif_set)): expected $expected_count, got $actual_count")
        end
    end

    return subset_motif_counts
end


"""
Test the EpicHyperSketch pipeline with the large example.
"""
function test_large_example(; use_gpu=true, min_counts=[5, 8, 10, 15])
    println("Creating large test dictionary...")
    test_dict = create_large_test_dict()
    println("Created dictionary with $(length(test_dict)) sequences")
    
    # Check CUDA availability if GPU requested
    if use_gpu && !CUDA.functional()
        @warn "CUDA not available, falling back to CPU"
        use_gpu = false
    end
    
    println("Backend: $(use_gpu ? "GPU (CUDA)" : "CPU")")
    
    # Verify ground truth (both consecutive and subset-based)
    subset_counts = verify_ground_truth(test_dict, 3; check_subsets=true)
    
    # Analyze enrichment distribution
    all_counts = collect(values(subset_counts))
    sorted_counts = sort(all_counts, rev=true)
    
    println("\n=== ENRICHMENT ANALYSIS ===")
    println("Total unique motifs: $(length(subset_counts))")
    println("Count distribution:")
    println("  Max count: $(maximum(all_counts))")
    println("  95th percentile: $(sorted_counts[max(1, div(length(sorted_counts) * 5, 100))])")
    println("  90th percentile: $(sorted_counts[max(1, div(length(sorted_counts) * 10, 100))])")
    println("  Median count: $(sorted_counts[max(1, div(length(sorted_counts), 2))])")
    println("  Motifs with count ≥ 5: $(count(x -> x >= 5, all_counts))")
    println("  Motifs with count ≥ 8: $(count(x -> x >= 8, all_counts))")
    println("  Motifs with count ≥ 10: $(count(x -> x >= 10, all_counts))")
    println("  Motifs with count ≥ 15: $(count(x -> x >= 15, all_counts))")
    
    # Test with EpicHyperSketch
    println("\n=== TESTING WITH EPICHYPERSKETCH ===")
    
    # Test with different min_count thresholds
    for min_count in min_counts
        println("\n--- Testing with min_count = $min_count ---")
        
        config = default_config(min_count=min_count, use_cuda=use_gpu)
        
        # Use appropriate function based on backend
        if use_gpu
            result = obtain_enriched_configurations(test_dict; 
                                                  motif_size=3, 
                                                  config=config,
                                                  filter_len=filter_len)
        else
            result = obtain_enriched_configurations_cpu(test_dict; 
                                                       motif_size=3, 
                                                       config=config,
                                                       filter_len=filter_len)
        end
        
        println("Found $(nrow(result)) enriched motifs:")
        for (i, row) in enumerate(eachrow(result))
            # Extract motif columns (m1, m2, m3)
            motif = [row.m1, row.m2, row.m3]
            # Check both consecutive and subset counts
            subset_count = get(subset_counts, Set(motif), 0)
            println("$i. $motif (subset: $subset_count)")
        end
        
        # Validate expected motifs are found when min_count is appropriate
        all_expected_motifs = [
            (Set([7, 19, 42]), 25),
            (Set([13, 28, 55]), 15), 
            (Set([22, 8, 39]), 12),
            (Set([3, 41, 67]), 8)
        ]
        
        # Filter expected motifs based on current min_count threshold
        expected_motifs_for_threshold = [motif_set for (motif_set, count) in all_expected_motifs if count >= min_count]
        
        found_expected = 0
        missing_expected = []
        
        for expected in expected_motifs_for_threshold
            if any(eachrow(result)) do row
                Set([row.m1, row.m2, row.m3]) == expected
            end
                found_expected += 1
            else
                missing_expected = [missing_expected..., expected]
            end
        end
        
        println("Expected enriched motifs (≥ $min_count): $(length(expected_motifs_for_threshold))")
        println("Found expected motifs: $found_expected / $(length(expected_motifs_for_threshold))")
        
        if !isempty(missing_expected)
            println("Missing expected motifs:")
            for missing in missing_expected
                actual_count = get(subset_counts, missing, 0)
                println("  - $(collect(missing)) (count: $actual_count)")
            end
        end
    end
end

"""
CPU-only version for CI/testing environments without GPU.
"""
function test_large_example_cpu()
    return test_large_example(use_gpu=false, min_counts=[8, 15])  # Fewer tests for CI
end

# Run the test if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Check environment variable to determine which test to run
    use_gpu = get(ENV, "EPIC_HYPERSKETCH_GPU_TESTS", "false") == "true"
    if use_gpu
        test_large_example()  # Full GPU test
    else
        test_large_example_cpu()  # CPU-only test
    end
end
