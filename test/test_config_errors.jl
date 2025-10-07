using EpicHyperSketch
using Test

@testset "Configuration and Error Handling" begin
    
    @testset "HyperSketchConfig validation" begin
        # Valid config
        config = HyperSketchConfig()
        @test config.delta == EpicHyperSketch.DEFAULT_CMS_DELTA
        @test config.use_cuda == true
        
        # Invalid parameters
        @test_throws EpicHyperSketch.InvalidConfigurationError HyperSketchConfig(delta=1.5)
        @test_throws EpicHyperSketch.InvalidConfigurationError HyperSketchConfig(epsilon=0.0)
        @test_throws EpicHyperSketch.InvalidConfigurationError HyperSketchConfig(min_count=0)
        @test_throws AssertionError HyperSketchConfig(batch_size=0)
    end
    
    @testset "Activation dictionary validation" begin
        # Valid dictionaries
        ordinary_dict = Dict(1 => [(feature=10, contribution=1.0f0), (feature=20, contribution=0.5f0)], 
                           2 => [(feature=15, contribution=0.8f0), (feature=25, contribution=1.2f0)])
        conv_dict = Dict(1 => [(filter=1, position=5, contribution=1.0f0)], 
                        2 => [(filter=2, position=10, contribution=0.9f0)])
        
        @test_nowarn EpicHyperSketch.validate_activation_dict(ordinary_dict)
        @test_nowarn EpicHyperSketch.validate_activation_dict(conv_dict)
        
        # Invalid dictionaries
        empty_dict = Dict{Int, Vector{EpicHyperSketch.OrdinaryFeature}}()
        @test_throws EpicHyperSketch.InvalidConfigurationError EpicHyperSketch.validate_activation_dict(empty_dict)
    end
    
    @testset "Type definitions" begin
        @test EpicHyperSketch.OrdinaryFeature == NamedTuple{(:feature, :contribution), Tuple{Int, EpicHyperSketch.FloatType}}
        @test EpicHyperSketch.ConvolutionFeature == NamedTuple{(:filter, :position, :contribution), Tuple{Int, Int, EpicHyperSketch.FloatType}}
        
        # Enum conversions
        @test EpicHyperSketch.symbol_to_case(:OrdinaryFeatures) == EpicHyperSketch.OrdinaryFeatures
        @test EpicHyperSketch.case_to_symbol(EpicHyperSketch.Convolution) == :Convolution
    end
end

@testset "Performance utilities" begin
    @testset "Memory monitoring" begin
        mem_info = EpicHyperSketch.memory_info()
        if CUDA.functional()
            @test haskey(mem_info, :used)
            @test haskey(mem_info, :free) 
            @test haskey(mem_info, :total)
        else
            @test mem_info === nothing
        end
    end
    
    @testset "Kernel parameter optimization" begin
        threads_1d = EpicHyperSketch.optimize_kernel_params(1000, (128,))
        @test length(threads_1d) == 1
        @test threads_1d[1] >= 32
        
        threads_2d = EpicHyperSketch.optimize_kernel_params(1000, (16, 16))
        @test length(threads_2d) == 2
    end
end