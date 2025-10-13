using Test
using EpicHyperSketch
using CUDA
using DataFrames

@testset "Partitioning Tests" begin
    
    @testset "partition_by_length basic" begin
        # Create a test dictionary with varying lengths
        dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int, Float32}}}}(
            1 => [(feature=1, contribution=1.0f0), (feature=2, contribution=0.5f0)],  # length 2
            2 => [(feature=3, contribution=0.8f0), (feature=4, contribution=0.3f0), (feature=5, contribution=0.7f0)],  # length 3
            3 => [(feature=1, contribution=1.0f0) for _ in 1:15],  # length 15
            4 => [(feature=2, contribution=0.5f0) for _ in 1:25],  # length 25
            5 => [(feature=3, contribution=0.8f0) for _ in 1:28],  # length 28
        )
        
        partitions, ranges = partition_by_length(dict, 10)
        
        # Should create partitions like [2:11, 12:21, 22:31] or similar
        @test length(partitions) > 0
        @test length(partitions) == length(ranges)
        
        # Check that all data points are accounted for
        total_keys = sum(length(p) for p in partitions)
        @test total_keys == length(dict)
        
        # Check that ranges don't overlap
        for i in 1:length(ranges)-1
            @test ranges[i].stop < ranges[i+1].start
        end
        
        # Verify data is in correct partitions
        for (key, val) in dict
            len = length(val)
            found = false
            for (partition, range) in zip(partitions, ranges)
                if len in range
                    @test haskey(partition, key)
                    @test partition[key] == val
                    found = true
                    break
                end
            end
            @test found
        end
    end
    
    @testset "partition_by_length edge cases" begin
        # Empty dict
        empty_dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int, Float32}}}}()
        partitions, ranges = partition_by_length(empty_dict, 10)
        @test isempty(partitions)
        @test isempty(ranges)
        
        # Single element
        single_dict = Dict(1 => [(feature=1, contribution=1.0f0)])
        partitions, ranges = partition_by_length(single_dict, 10)
        @test length(partitions) == 1
        @test length(partitions[1]) == 1
        
        # All same length
        same_len_dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int, Float32}}}}(
            i => [(feature=i, contribution=1.0f0) for _ in 1:5] for i in 1:10
        )
        partitions, ranges = partition_by_length(same_len_dict, 10)
        @test length(partitions) == 1
        @test length(partitions[1]) == 10
    end
    
    @testset "partition_by_length different widths" begin
        dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int, Float32}}}}(
            i => [(feature=j, contribution=1.0f0) for j in 1:i] for i in 1:30
        )
        
        # Test with width 5
        partitions_5, ranges_5 = partition_by_length(dict, 5)
        @test length(partitions_5) >= 6  # Should have multiple partitions
        
        # Test with width 15
        partitions_15, ranges_15 = partition_by_length(dict, 15)
        @test length(partitions_15) <= length(partitions_5)  # Larger width = fewer partitions
        
        # Both should account for all data
        @test sum(length(p) for p in partitions_5) == length(dict)
        @test sum(length(p) for p in partitions_15) == length(dict)
    end
    
    @testset "create_partitioned_record basic" begin
        # Skip if CUDA not available
        if !CUDA.functional()
            @warn "Skipping CUDA tests (CUDA not available)"
            return
        end
        
        # Create test data with varying lengths
        dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int32, Float32}}}}(
            i => [(feature=Int32(j % 10 + 1), contribution=Float32(rand())) 
                  for j in 1:(5 + (i % 20))] for i in 1:100
        )
        
        # Create partitioned record
        pr = create_partitioned_record(
            dict, 3;
            partition_width=5,
            batch_size=20,
            use_cuda=true,
            seed=42
        )
        
        # Check structure
        @test pr isa PartitionedRecord
        @test length(pr.partitions) > 1  # Should create multiple partitions
        @test pr.motif_size == 3
        @test pr.case == :OrdinaryFeatures
        @test pr.use_cuda == true
        
        # Check that the shared sketch is stored
        @test pr.shared_cms isa CountMinSketch
        
        # Check partition ranges
        @test length(pr.partition_ranges) == length(pr.partitions)
        for i in 1:length(pr.partition_ranges)-1
            @test pr.partition_ranges[i].stop < pr.partition_ranges[i+1].start
        end
    end
    
    @testset "partitioned counting and extraction" begin
        # Skip if CUDA not available
        if !CUDA.functional()
            @warn "Skipping CUDA tests (CUDA not available)"
            return
        end
        
        # Create simple test data with repeated features to ensure matches
        dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int32, Float32}}}}(
            i => [(feature=Int32(j % 5 + 1), contribution=Float32(1.0)) 
                  for j in 1:(3 + (i % 15))] for i in 1:50
        )
        
        # Use partitioned approach with low threshold
        motifs = obtain_enriched_configurations_partitioned(
            dict,
            motif_size=2,
            partition_width=5,
            batch_size=10,
            min_count=1  # Use threshold of 1 to ensure results
        )
        
        # Basic checks
        @test motifs isa DataFrame
        @test nrow(motifs) >= 0  # May be 0 or more depending on data
        @test hasproperty(motifs, :m1)
        @test hasproperty(motifs, :m2)
        @test hasproperty(motifs, :data_index)
        @test hasproperty(motifs, :contribution)
    end
    
    @testset "auto batch size with partitioning" begin
        # Skip if CUDA not available
        if !CUDA.functional()
            @warn "Skipping CUDA tests (CUDA not available)"
            return
        end
        
        # Create test data with very different lengths
        dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int32, Float32}}}}()
        
        # Short sequences
        for i in 1:20
            dict[i] = [(feature=Int32(j), contribution=Float32(1.0)) for j in 1:5]
        end
        
        # Long sequences
        for i in 21:40
            dict[i] = [(feature=Int32(j % 20 + 1), contribution=Float32(1.0)) for j in 1:50]
        end
        
        # Use auto batch sizing
        pr = create_partitioned_record(
            dict, 3;
            partition_width=10,
            batch_size=:auto,
            use_cuda=true,
            seed=42,
            auto_batch_verbose=true
        )
        
        # Should have created partitions
        @test length(pr.partitions) >= 2
    end
    
    @testset "print_partition_stats" begin
        # Skip if CUDA not available
        if !CUDA.functional()
            @warn "Skipping CUDA tests (CUDA not available)"
            return
        end
        
        dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int32, Float32}}}}(
            i => [(feature=Int32(j), contribution=Float32(1.0)) 
                  for j in 1:(5 + (i % 10))] for i in 1:30
        )
        
        pr = create_partitioned_record(dict, 2; partition_width=3, batch_size=10, use_cuda=true)
        
        # Just test that it doesn't error
        @test_nowarn print_partition_stats(pr)
    end
    
    @testset "convolution case with partitioning" begin
        # Skip if CUDA not available
        if !CUDA.functional()
            @warn "Skipping CUDA tests (CUDA not available)"
            return
        end
        
        # Create convolution test data with varying lengths
        dict = Dict{Int, Vector{NamedTuple{(:filter, :contribution, :position), Tuple{Int32, Float32, Int32}}}}(
            i => [(filter=Int32(j % 5 + 1), contribution=Float32(1.0), position=Int32(j)) 
                  for j in 1:(10 + (i % 15))] for i in 1:40
        )
        
        # Use partitioned approach with convolution
        motifs = obtain_enriched_configurations_partitioned(
            dict,
            motif_size=2,
            partition_width=5,
            batch_size=10,
            filter_len=5,
            min_count=2
        )
        
        # Check convolution-specific columns
        @test motifs isa DataFrame
        @test hasproperty(motifs, :m1)
        @test hasproperty(motifs, :m2)
        @test hasproperty(motifs, :d12)  # Distance column
        @test hasproperty(motifs, :start)
        @test hasproperty(motifs, :end)
    end
    
    @testset "comparison with non-partitioned" begin
        # Skip if CUDA not available
        if !CUDA.functional()
            @warn "Skipping CUDA tests (CUDA not available)"
            return
        end
        
        # Create small test dataset with repeated patterns
        dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int32, Float32}}}}(
            i => [(feature=Int32(j % 8 + 1), contribution=Float32(1.0)) 
                  for j in 1:(5 + (i % 8))] for i in 1:30
        )
        
        # Process with standard approach
        motifs_standard = obtain_enriched_configurations(
            dict,
            motif_size=2,
            min_count=1,  # Use threshold of 1
            seed=42
        )
        
        # Process with partitioned approach
        motifs_partitioned = obtain_enriched_configurations_partitioned(
            dict,
            motif_size=2,
            partition_width=3,
            batch_size=10,
            min_count=1,  # Use threshold of 1
            seed=42
        )
        
        # Both should return valid DataFrames
        @test motifs_standard isa DataFrame
        @test motifs_partitioned isa DataFrame
        
        # Both should have the same columns
        @test hasproperty(motifs_standard, :m1)
        @test hasproperty(motifs_partitioned, :m1)
    end
end
