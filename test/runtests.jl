using EpicHyperSketch
using Test
using IterTools

@testset "EpicHyperSketch.jl" begin
        
    @testset "CountMinSketch constructor (CPU only)" begin
        IntType = EpicHyperSketch.IntType
        motif_size = 5
        delta = 0.001
        epsilon = 0.0001
        use_cuda = false
        cms = EpicHyperSketch.CountMinSketch(
            motif_size; delta=delta, epsilon=epsilon, use_cuda=use_cuda)

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


    @testset "Record construction test set 1" begin
        E = EpicHyperSketch

        """
        for :Convolution case, need to test the non-zero portion of 
            vecRefArray[i][:, POSITION_COLUMN, n] is sorted in ascending order for all i, n
                i is the batch index, 
                n is the within-batch index
        """

        activation_dict = Dict(
         1=>[(filter=2, position=5), (filter=2, position=10), (filter=3, position=15)], 
         2=>[(filter=1, position=15), (filter=3, position=25)], 
         3=>[(filter=2, position=1), (filter=2, position=6), (filter=3, position=12)],
         4=>[(filter=1, position=3), (filter=2, position=8), (filter=4, position=20)],
         5=>[(filter=3, position=7), (filter=4, position=14)],
        )

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


end
