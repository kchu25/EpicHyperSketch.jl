using EpicHyperSketch
using CUDA
using Test
using DataFrames

@testset "Memory Management and Auto Batch Size" begin
    # Create test data
    activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()
    filter_len = 8
    
    # Create 100 sequences with moderate complexity
    for i in 1:100
        features = [
            (filter=1, contribution=1.0f0, position=10),
            (filter=2, contribution=1.0f0, position=20),
            (filter=3, contribution=1.0f0, position=35),
            (filter=4, contribution=1.0f0, position=50)
        ]
        activation_dict[i] = features
    end
    
    max_active_len = EpicHyperSketch.get_max_active_len(activation_dict)
    motif_size = 3
    case = :Convolution
    
    @testset "Memory estimation functions" begin
        # Test per-batch memory estimation
        per_batch_mem = EpicHyperSketch.estimate_memory_per_batch(max_active_len, motif_size, case)
        @test per_batch_mem > 0
        @test isa(per_batch_mem, Real)
        
        # Test fixed memory estimation
        fixed_mem = EpicHyperSketch.estimate_fixed_memory(max_active_len, motif_size, case)
        @test fixed_mem > 0
        @test isa(fixed_mem, Real)
        
        println("  Memory per data point: $(round(per_batch_mem/1024, digits=2)) KB")
        println("  Fixed memory: $(round(fixed_mem/1024^2, digits=2)) MB")
    end
    
    @testset "Optimal batch size calculation" begin
        # Test with explicit memory budget
        batch_size = EpicHyperSketch.calculate_optimal_batch_size(
            100, max_active_len, motif_size, case;
            target_memory_gb=1.0,
            use_cuda=false
        )
        @test batch_size > 0
        @test batch_size <= 100
        println("  Calculated batch_size for 1GB: $batch_size")
        
        # Test with auto-detection (CPU)
        batch_size_auto = EpicHyperSketch.calculate_optimal_batch_size(
            100, max_active_len, motif_size, case;
            use_cuda=false
        )
        @test batch_size_auto > 0
        @test batch_size_auto <= 100
        println("  Auto batch_size (CPU): $batch_size_auto")
        
        # Test with GPU if available
        if CUDA.functional()
            batch_size_gpu = EpicHyperSketch.calculate_optimal_batch_size(
                100, max_active_len, motif_size, case;
                use_cuda=true
            )
            @test batch_size_gpu > 0
            println("  Auto batch_size (GPU): $batch_size_gpu")
        end
    end
    
    @testset "Auto-configure batch size" begin
        result = EpicHyperSketch.auto_configure_batch_size(
            activation_dict, motif_size, case;
            use_cuda=false,
            verbose=false
        )
        
        @test haskey(result, :batch_size)
        @test haskey(result, :num_batches)
        @test haskey(result, :memory_per_batch_mb)
        @test result.batch_size > 0
        @test result.num_batches > 0
        @test result.total_data_points == 100
        
        println("  Batch size: $(result.batch_size)")
        println("  Num batches: $(result.num_batches)")
        println("  Memory per batch: $(round(result.memory_per_batch_mb, digits=2)) MB")
        println("  Peak memory estimate: $(round(result.estimated_peak_memory_gb, digits=3)) GB")
    end
    
    @testset "Integration with Record constructor" begin
        # Test with :auto batch size
        record = EpicHyperSketch.Record(
            activation_dict, motif_size;
            batch_size=:auto,
            use_cuda=false,
            filter_len=filter_len,
            auto_batch_verbose=false
        )
        
        @test isa(record, EpicHyperSketch.Record)
        @test EpicHyperSketch.num_batches(record) > 0
        
        # Test with explicit batch size
        record2 = EpicHyperSketch.Record(
            activation_dict, motif_size;
            batch_size=25,
            use_cuda=false,
            filter_len=filter_len
        )
        
        @test isa(record2, EpicHyperSketch.Record)
        @test EpicHyperSketch.num_batches(record2) == 4  # 100 / 25 = 4
    end
    
    @testset "Memory report printing" begin
        config = EpicHyperSketch.default_config(min_count=5, use_cuda=false, batch_size=50)
        
        # Should not throw an error
        @test_nowarn EpicHyperSketch.print_memory_report(
            activation_dict, motif_size, case, config
        )
    end
    
    @testset "Edge cases" begin
        # Very small dataset
        small_dict = Dict(1 => [(filter=1, contribution=1.0f0, position=5)])
        result_small = EpicHyperSketch.auto_configure_batch_size(
            small_dict, 1, :Convolution;
            use_cuda=false
        )
        @test result_small.batch_size >= 1
        @test result_small.num_batches == 1
        
        # Test with insufficient memory (should error)
        @test_throws ErrorException EpicHyperSketch.calculate_optimal_batch_size(
            1000, 100, 5, :Convolution;
            target_memory_gb=0.001,  # Too small
            use_cuda=false
        )
    end
    
    @testset "Integration with obtain_enriched_configurations" begin
        # Test with auto batch size
        config_auto = EpicHyperSketch.default_config(
            min_count=1,
            batch_size=:auto,
            use_cuda=false
        )
        
        result = EpicHyperSketch.obtain_enriched_configurations_cpu(
            activation_dict;
            motif_size=motif_size,
            filter_len=filter_len,
            config=config_auto
        )
        
        @test isa(result, DataFrames.DataFrame)
        @test nrow(result) > 0
    end
end

println("\n✓ All memory management tests passed!")
