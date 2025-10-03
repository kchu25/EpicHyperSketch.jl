using EpicHyperSketch
using Test

@testset "CPU Implementation (Both Cases)" begin
    
    @testset "CPU vs GPU comparison (small dataset)" begin
        # Create a small test dataset
        activation_dict_conv = Dict(
            1 => [(filter=2, position=5), (filter=3, position=15)], 
            2 => [(filter=1, position=10), (filter=4, position=25)], 
            3 => [(filter=2, position=1), (filter=3, position=12)]
        )
        
        motif_size = 2
        filter_len = 8
        min_count = 1
        
        # Test CPU version
        @info "Testing CPU implementation..."
        config_cpu = EpicHyperSketch.default_config(min_count=min_count, use_cuda=false)
        
        result_cpu = EpicHyperSketch.obtain_enriched_configurations_cpu(
            activation_dict_conv;
            motif_size=motif_size,
            filter_len=filter_len,
            min_count=min_count,
            config=config_cpu
        )
        
        @test isa(result_cpu, Set)
        @info "CPU result: $(result_cpu)"
        @info "Number of configurations found (CPU): $(length(result_cpu))"
        
        # If CUDA is available, compare with GPU version
        if CUDA.functional()
            @info "CUDA available, comparing with GPU implementation..."
            config_gpu = EpicHyperSketch.default_config(min_count=min_count, use_cuda=true)
            
            result_gpu = EpicHyperSketch.obtain_enriched_configurations(
                activation_dict_conv;
                motif_size=motif_size,
                filter_len=filter_len,
                min_count=min_count,
                config=config_gpu
            )
            
            @info "GPU result: $(result_gpu)"
            @info "Number of configurations found (GPU): $(length(result_gpu))"
            
            # Results should be identical
            @test result_cpu == result_gpu
            @info "✓ CPU and GPU results match!"
        else
            @info "CUDA not available, skipping GPU comparison"
        end
    end
    
    @testset "CPU ordinary features" begin
        # Test ordinary features case
        activation_dict_ord = Dict(
            1 => [10, 20, 30],
            2 => [15, 25],
            3 => [5, 35, 40]
        )
        
        motif_size = 2
        min_count = 1
        
        config_cpu = EpicHyperSketch.default_config(min_count=min_count, use_cuda=false)
        
        result_cpu = EpicHyperSketch.obtain_enriched_configurations_cpu(
            activation_dict_ord;
            motif_size=motif_size,
            filter_len=nothing,
            min_count=min_count,
            config=config_cpu
        )
        
        @test isa(result_cpu, Set)
        @info "CPU ordinary result: $(result_cpu)"
        @info "Number of ordinary configurations found (CPU): $(length(result_cpu))"
    end
end