#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using EpicHyperSketch
using BenchmarkTools
using CUDA
using Random

println("Performance Benchmark - Before GPU Optimization")
println("=" ^ 50)

# Test data setup - use the existing large test
include("test_large_example_convolution.jl")

filter_len = 8
motif_size = 3

# Use the existing test data function
activation_dict = create_large_convolution_test_dict()

println("Test setup:")
println("  Number of sequences: $(length(activation_dict))")
println("  Filter length: $filter_len")
println("  Motif size: $motif_size")
println()

# Create configs
config_cpu = EpicHyperSketch.default_config(min_count=5, use_cuda=false)
config_gpu = EpicHyperSketch.default_config(min_count=5, use_cuda=true)

# CPU Benchmark
println("CPU Performance:")
cpu_time = @benchmark EpicHyperSketch.obtain_enriched_configurations_cpu(
    $activation_dict; motif_size=$motif_size, filter_len=$filter_len, 
    min_count=5, config=$config_cpu
) samples=5 seconds=30

println("  Median time: $(BenchmarkTools.prettytime(median(cpu_time).time))")
println("  Memory: $(BenchmarkTools.prettymemory(median(cpu_time).memory))")
println()

# GPU Benchmark (if available)
if CUDA.functional()
    println("GPU Performance:")
    gpu_time = @benchmark CUDA.@sync EpicHyperSketch.obtain_enriched_configurations(
        $activation_dict; motif_size=$motif_size, filter_len=$filter_len, config=$config_gpu
    ) samples=5 seconds=30
    
    println("  Median time: $(BenchmarkTools.prettytime(median(gpu_time).time))")
    println("  Memory: $(BenchmarkTools.prettymemory(median(gpu_time).memory))")
    println("  Speedup: $(median(cpu_time).time / median(gpu_time).time)x")
else
    println("GPU not available")
end

println()
println("Benchmark completed!")
