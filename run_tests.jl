#!/usr/bin/env julia

"""
Test runner script for EpicHyperSketch with different configurations.

Usage:
    julia run_tests.jl [--gpu] [--large] [--all]

Options:
    --gpu     Enable GPU tests (requires CUDA)
    --large   Enable large-scale tests  
    --all     Enable both GPU and large tests
    (no args) Run only basic CPU tests (GitHub Actions compatible)
"""

using Pkg
Pkg.activate(".")

# Parse command line arguments
args = ARGS
run_gpu = "--gpu" in args || "--all" in args
run_large = "--large" in args || "--all" in args

# Set environment variables
if run_gpu
    ENV["EPIC_HYPERSKETCH_GPU_TESTS"] = "true"
    println("✓ GPU tests enabled")
end

if run_large
    ENV["EPIC_HYPERSKETCH_LARGE_TESTS"] = "true"  
    println("✓ Large-scale tests enabled")
end

if !run_gpu && !run_large
    println("✓ Running basic CPU tests only (GitHub Actions mode)")
end

# Run the tests
println("\n" * "="^60)
println("Starting EpicHyperSketch test suite...")
println("="^60)

using Pkg
Pkg.test()

println("\n" * "="^60)
println("Test suite completed!")
println("="^60)
