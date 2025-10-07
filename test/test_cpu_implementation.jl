using EpicHyperSketch
using Test

@testset "CPU Implementation (Both Cases)" begin
    @testset "CPU Convolution - Large Dataset (2000 points)" begin
        # Generate 2000 data points with implicit ground truth pattern
        # Pattern: Every 10th data point has (filter=1, position=5) and (filter=2, position=15)
        # This creates a known motif that should appear ~200 times
        n_points = 2000
        motif_size = 2
        filter_len = 8
        
        activation_dict_conv = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()
        
        # Generate data with implicit pattern
        for i in 1:n_points
            features = EpicHyperSketch.ConvolutionFeature[]
            
            # Every 10th point contains the ground truth motif
            if i % 10 == 0
                push!(features, (filter=1, contribution=1.0f0, position=5))
                push!(features, (filter=2, contribution=1.0f0, position=15))
            end
            
            # Add some random noise features
            num_noise = rand(1:4)
            for _ in 1:num_noise
                push!(features, (filter=rand(3:6), contribution=rand(Float32), position=rand(1:30)))
            end
            
            activation_dict_conv[i] = features
        end
        
        @info "Generated $(n_points) convolution data points"
        @info "Expected ground truth: filters [1,2] with distance ~$(15-5-filter_len) should be highly enriched"
        
        # Test CPU version
        config_cpu = EpicHyperSketch.default_config(min_count=50, use_cuda=false)
        
        result_cpu = EpicHyperSketch.obtain_enriched_configurations_cpu(
            activation_dict_conv;
            motif_size=motif_size,
            filter_len=filter_len,
            min_count=50,
            config=config_cpu
        )
        
        @test isa(result_cpu, Set)
        @test length(result_cpu) > 0  # Should find at least the ground truth pattern
        @info "Number of enriched configurations found (CPU): $(length(result_cpu))"
        
        # Check if ground truth pattern is found
        # For filters [1,2] with positions 5 and 15, distance = 15-5-filter_len = 2
        expected_distance = 15 - 5 - filter_len
        found_ground_truth = any(cfg -> cfg[1] == 1 && cfg[2] == 2 && cfg[3] == expected_distance, result_cpu)
        
        if found_ground_truth
            @info "✓ Ground truth pattern found: filters [1,2] with distance $(expected_distance)"
        else
            @info "Available configurations: $(result_cpu)"
        end
    end
    
    @testset "CPU Ordinary Features - Large Dataset (2000 points)" begin
        # Generate 2000 data points with implicit ground truth pattern
        # Pattern: Every 8th data point has features [5, 15]
        # This creates a known motif that should appear ~250 times
        n_points = 2000
        motif_size = 2
        
        activation_dict_ord = Dict{Int, Vector{EpicHyperSketch.OrdinaryFeature}}()
        
        # Generate data with implicit pattern
        for i in 1:n_points
            features = EpicHyperSketch.OrdinaryFeature[]
            
            # Every 8th point contains the ground truth motif
            if i % 8 == 0
                push!(features, (feature=5, contribution=1.0f0))
                push!(features, (feature=15, contribution=1.0f0))
            end
            
            # Add some random noise features
            num_noise = rand(1:5)
            for _ in 1:num_noise
                push!(features, (feature=rand(1:30), contribution=rand(Float32)))
            end
            
            # Sort by feature ID and aggregate contributions for duplicates
            unique_features = Dict{Int, Float32}()
            for feat in features
                unique_features[feat.feature] = get(unique_features, feat.feature, 0.0f0) + feat.contribution
            end
            activation_dict_ord[i] = [(feature=k, contribution=v) for (k, v) in sort(collect(unique_features))]
        end
        
        @info "Generated $(n_points) ordinary feature data points"
        @info "Expected ground truth: features [5,15] should be highly enriched (~250 occurrences)"
        
        # Test CPU version
        config_cpu = EpicHyperSketch.default_config(min_count=80, use_cuda=false)
        
        result_cpu = EpicHyperSketch.obtain_enriched_configurations_cpu(
            activation_dict_ord;
            motif_size=motif_size,
            filter_len=nothing,
            min_count=80,
            config=config_cpu
        )
        
        @test isa(result_cpu, Set)
        @test length(result_cpu) > 0  # Should find at least the ground truth pattern
        @info "Number of enriched configurations found (CPU): $(length(result_cpu))"
        
        # Check if ground truth pattern is found
        found_ground_truth = any(cfg -> cfg == (5, 15), result_cpu)
        
        if found_ground_truth
            @info "✓ Ground truth pattern found: features [5,15]"
        else
            @info "Available configurations: $(result_cpu)"
        end
    end
end