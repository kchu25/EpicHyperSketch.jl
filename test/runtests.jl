using EpicHyperSketch
using Test
using IterTools
using Random

# include("test_config_errors.jl")# Large tests with CPU backend
include("test_cpu_implementation.jl")

"""
# Full GPU tests  
EPIC_HYPERSKETCH_GPU_TESTS=true julia --project=. -e "using Pkg; Pkg.test()"
"""

# Check if we should run GPU tests (local development) or CPU only (CI/GitHub Actions)
const RUN_GPU_TESTS = get(ENV, "EPIC_HYPERSKETCH_GPU_TESTS", "false") == "true"

RUN_GPU_TESTS && include("test_large_example_ordinary.jl")

@testset "EpicHyperSketch.jl" begin
       
    @testset "CountMinSketch constructor (CPU only)" begin
        IntType = EpicHyperSketch.IntType
        motif_size = 5
        delta = 0.001
        epsilon = 0.0001
        use_cuda = false
        cms = EpicHyperSketch.CountMinSketch(
            motif_size; delta=delta, epsilon=epsilon, 
                use_cuda=use_cuda)

        println("hash_coeffs size: ", size(cms.hash_coeffs))
        println("sketch size: ", size(cms.sketch))

        @test isa(cms, CountMinSketch)
        @test isa(cms.hash_coeffs, AbstractMatrix{IntType})
        @test isa(cms.sketch, AbstractMatrix{IntType})
        rows = EpicHyperSketch.cms_rows(delta)
        num_counters = EpicHyperSketch.cms_num_counters(rows, epsilon)
        cols = EpicHyperSketch.cms_cols(num_counters, rows)
        @test size(cms.hash_coeffs) == (rows, motif_size)
        @test size(cms.sketch) == (rows, cols)
    end


    activation_dict = Dict(
         1=>[(filter=2, contribution=1.0f0, position=5), (filter=2, contribution=1.0f0, position=10), (filter=3, contribution=1.0f0, position=15)], 
         2=>[(filter=1, contribution=1.0f0, position=15), (filter=3, contribution=1.0f0, position=25)], 
         3=>[(filter=2, contribution=1.0f0, position=1), (filter=2, contribution=1.0f0, position=6), (filter=3, contribution=1.0f0, position=12)],
         4=>[(filter=1, contribution=1.0f0, position=3), (filter=2, contribution=1.0f0, position=8), (filter=4, contribution=1.0f0, position=20)],
         5=>[(filter=3, contribution=1.0f0, position=7), (filter=4, contribution=1.0f0, position=14)],
        )

    @testset "Record construction test set 1" begin
        E = EpicHyperSketch

        """
        for :Convolution case, need to test the non-zero portion of 
            vecRefArray[i][:, POSITION_COLUMN, n] is sorted in ascending order for all i, n
                i is the batch index, 
                n is the within-batch index
        """

        
        motif_size = 2; batch_size = 2; use_cuda = false
        E.sort_activation_dict!(activation_dict)

        # determine case and max_active_length
        case = E.dict_case(activation_dict)
        @test case == :Convolution
        max_active_len = E.get_max_active_len(activation_dict)
        @test max_active_len == 3
        vecRefArray = E.constructVecRefArrays(activation_dict, max_active_len; 
                batch_size=batch_size, case=case, use_cuda=use_cuda)
        @test length(vecRefArray) == 3  # number of batches
        @test size(vecRefArray[1]) == (3, 2, 2)
        # test each first consecutive nonzero elements in refArray[:,2,n] in vecRefArray is sorted in ascending order
        # @info "$(vecRefArray[1][:, E.POSITION_COLUMN, 1])"
        for i in eachindex(vecRefArray)
            b_size_here = size(vecRefArray[i], 3)
            for j in 1:b_size_here
                col = @view vecRefArray[i][:, E.POSITION_COLUMN, j]
                col_filtered = takewhile(x -> x != 0, col) # first consecutive nonzero elements
                @test issorted(collect(col_filtered))
            end
        end
    end

    @testset "Record constructor test (CPU only)" begin
        # Test data for Record construction
        # activation_dict_ordinary = Dict(
        #     1 => [10, 20, 30],
        #     2 => [15, 25],
        #     3 => [5, 35, 40, 45]
        # )
        
        # activation_dict_convolution = Dict(
        #     1 => [(filter=2, position=5), (filter=2, position=10), (filter=3, position=15)], 
        #     2 => [(filter=1, position=15), (filter=3, position=25)], 
        #     3 => [(filter=2, position=1), (filter=2, position=6), (filter=3, position=12)]
        # )

        # # Test Record construction with ordinary features (CPU only)
        # motif_size = 2
        # batch_size = 2
        # use_cuda = false
        # filter_len = nothing
        
        # record_ordinary = Record(activation_dict_ordinary, motif_size; 
        #                         batch_size=batch_size, use_cuda=use_cuda, filter_len=filter_len)
        
        # @test isa(record_ordinary, Record)
        # @test record_ordinary.motif_size == motif_size
        # @test record_ordinary.case == :OrdinaryFeatures
        # @test record_ordinary.filter_len == nothing
        # @test length(record_ordinary.vecRefArray) > 0
        # @test isa(record_ordinary.cms, CountMinSketch)
        # @test size(record_ordinary.combs, 1) == motif_size
        # @test length(record_ordinary.selectedCombs) == length(record_ordinary.vecRefArray)

        # # Test Record construction with convolution features (CPU only)  
        # filter_len = 8
        # record_convolution = Record(activation_dict_convolution, motif_size;
        #                            batch_size=batch_size, use_cuda=use_cuda, filter_len=filter_len)
        
        # @test isa(record_convolution, Record)
        # @test record_convolution.motif_size == motif_size
        # @test record_convolution.case == :Convolution
        # @test record_convolution.filter_len == filter_len
        # @test length(record_convolution.vecRefArray) > 0
        # @test isa(record_convolution.cms, CountMinSketch)
        # @test size(record_convolution.combs, 1) == motif_size
        # @test length(record_convolution.selectedCombs) == length(record_convolution.vecRefArray)
        
        # # Test num_batches function
        # @test num_batches(record_ordinary) == length(record_ordinary.vecRefArray)
        # @test num_batches(record_convolution) == length(record_convolution.vecRefArray)
    end

    if RUN_GPU_TESTS
        @testset "Large Example Tests OrdinaryFeature case (GPU)" begin
            println("Running full large example test with GPU backend...")
            # Only run if CUDA is available
            if CUDA.functional()
                test_large_example_ordinary()  # From test_large_example.jl
                @test true  # If we get here without error, test passes
            else
                @test_skip "CUDA not available for GPU tests"
            end
        end
    end

end
