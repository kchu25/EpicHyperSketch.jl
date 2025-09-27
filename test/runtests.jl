using EpicHyperSketch
using Test

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


end
