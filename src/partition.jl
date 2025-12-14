# Partitioning utilities for activation_dict by value length
# This enables efficient processing when sequences have varying lengths

"""
    PartitionedRecord

A structure for sequentially processing activation_dict partitions with a shared
CountMinSketch. Stores partition data but creates Records on-demand to minimize
memory usage.

# Fields
- `partitions::Vector{Dict}`: Activation dict partitions (each subset by length)
- `shared_cms::CountMinSketch`: The shared count-min sketch
- `partition_ranges::Vector{UnitRange{Int}}`: Length ranges for each partition
- `motif_size::IntType`: Size of motifs
- `case::Symbol`: Processing case (:OrdinaryFeatures or :Convolution)
- `filter_len::Union{IntType, Nothing}`: Filter length for convolution case
- `use_cuda::Bool`: Whether to use CUDA
- `batch_size::Union{Int, Symbol}`: Batch size (:auto or integer)
- `auto_batch_verbose::Bool`: Whether to print auto batch size details
"""
struct PartitionedRecord
    partitions::Vector{Dict}
    shared_cms::CountMinSketch
    partition_ranges::Vector{UnitRange{Int}}
    motif_size::IntType
    case::Symbol
    filter_len::Union{IntType, Nothing}
    use_cuda::Bool
    batch_size::Union{Int, Symbol}
    auto_batch_verbose::Bool
end

"""
    partition_by_length(activation_dict, partition_width=10)

Partition an activation dictionary into sub-dictionaries based on the length
of the feature vectors.

# Arguments
- `activation_dict`: Dictionary mapping data point IDs to feature vectors
- `partition_width`: Width of each length range (default: 10)

# Returns
- `partitions::Vector{Dict}`: Vector of sub-dictionaries, one per partition
- `ranges::Vector{UnitRange{Int}}`: Length ranges for each partition

# Example
```julia
partitions, ranges = partition_by_length(activation_dict, 10)
# ranges might be: [1:10, 11:20, 21:30, ...]
```
"""
function partition_by_length(
    activation_dict::Dict{T, Vector{S}},
    partition_width::Integer=10
) where {T <: Integer, S}
    
    if isempty(activation_dict)
        return Dict{T, Vector{S}}[], UnitRange{Int}[]
    end
    
    # Find the range of lengths
    lengths = [length(v) for v in values(activation_dict)]
    min_len = minimum(lengths)
    max_len = maximum(lengths)
    
    # Create partition ranges
    ranges = UnitRange{Int}[]
    current = min_len
    while current <= max_len
        range_end = min(current + partition_width - 1, max_len)
        push!(ranges, current:range_end)
        current = range_end + 1
    end
    
    # Create partition dictionaries
    partitions = [Dict{T, Vector{S}}() for _ in ranges]
    
    for (key, val) in activation_dict
        len = length(val)
        # Find which partition this belongs to
        for (i, range) in enumerate(ranges)
            if len in range
                partitions[i][key] = val
                break
            end
        end
    end
    
    # Filter out empty partitions
    non_empty_indices = findall(p -> !isempty(p), partitions)
    partitions = partitions[non_empty_indices]
    ranges = ranges[non_empty_indices]
    
    return partitions, ranges
end

"""
    create_partitioned_record(activation_dict, motif_size; kwargs...)

Create a PartitionedRecord that splits the activation_dict by value lengths
and shares a single CountMinSketch. Records are created on-demand during
processing to minimize memory usage.

# Arguments
- `activation_dict`: Dictionary mapping data point IDs to feature vectors
- `motif_size`: Number of features per motif
- `partition_width`: Width of length ranges for partitioning (default: 10)
- `batch_size`: Batch size per partition (can be :auto or Integer)
- `use_cuda`: Whether to use CUDA (default: true)
- `filter_len`: Filter length for convolution case
- `seed`: Random seed for sketch initialization
- `auto_batch_verbose`: Whether to print batch size info

# Returns
A PartitionedRecord that processes partitions sequentially.
"""
function create_partitioned_record(
    activation_dict::Dict{T, Vector{S}},
    motif_size::Integer;
    partition_width::Integer=10,
    batch_size::Union{Integer, Symbol}=BATCH_SIZE,
    use_cuda::Bool=true,
    filter_len::Union{Integer, Nothing}=nothing,
    seed::Union{Int, Nothing}=nothing,
    auto_batch_verbose::Bool=false,
    verbose::Bool=false
) where {T <: Integer, S}
    
    verbose && @info "Partitioning activation_dict by value lengths (width=$partition_width)..."
    
    # Partition the dictionary
    partitions, ranges = partition_by_length(activation_dict, partition_width)
    
    if isempty(partitions)
        error("Cannot create PartitionedRecord from empty activation_dict")
    end
    
    verbose && @info "Created $(length(partitions)) partitions: $ranges"
    
    # Determine case
    filter_empty!(activation_dict)
    case = dict_case(activation_dict)
    
    # Create a shared CountMinSketch
    shared_cms = CountMinSketch(motif_size; case=case, use_cuda=use_cuda, seed=seed)
    verbose && @info "Created shared CountMinSketch"
    
    return PartitionedRecord(
        partitions,
        shared_cms,
        ranges,
        IntType(motif_size),
        case,
        !isnothing(filter_len) ? IntType(filter_len) : nothing,
        use_cuda,
        batch_size,
        auto_batch_verbose
    )
end

"""
    _create_record_for_partition(partition, pr::PartitionedRecord)

Internal helper to create a Record for a single partition.
"""
function _create_record_for_partition(
    partition::Dict,
    pr::PartitionedRecord,
    partition_idx::Int,
    verbose::Bool=false
)
    # Preprocess partition
    filter_empty!(partition)
    sort_activation_dict!(partition, case=pr.case)
    max_active_len = get_max_active_len(partition)
    
    # Calculate batch size for this partition
    actual_batch_size = pr.batch_size
    if pr.batch_size == :auto
        result = auto_configure_batch_size(
            partition, pr.motif_size, pr.case; 
            use_cuda=pr.use_cuda, verbose=pr.auto_batch_verbose
        )
        actual_batch_size = result.batch_size
        verbose && @info "  Auto-configured batch_size: $actual_batch_size ($(result.num_batches) batches, ~$(round(result.estimated_peak_memory_gb, digits=2)) GB estimated)"
    else
        verbose && @info "  Using batch_size: $actual_batch_size"
    end
    
    # Generate combinations and reference arrays
    combs = generate_combinations(pr.motif_size, max_active_len; use_cuda=pr.use_cuda)
    vecRefArray, vecRefArrayContrib = constructVecRefArrays(
        partition, max_active_len; 
        batch_size=actual_batch_size, case=pr.case, use_cuda=pr.use_cuda
    )
    
    # Create selected combinations arrays
    selectedCombs = pr.use_cuda ?
        [CUDA.fill(false, (size(combs, 2), size(vecRefArray[j], 3))) for j in eachindex(vecRefArray)] :
        [fill(false, (size(combs, 2), size(vecRefArray[j], 3))) for j in eachindex(vecRefArray)]
    
    # Create Record with shared sketch
    return Record(
        vecRefArray,
        vecRefArrayContrib,
        combs,
        pr.shared_cms,  # Share the same sketch!
        selectedCombs,
        pr.case,
        pr.motif_size,
        pr.filter_len,
        pr.use_cuda
    )
end

"""
    obtain_enriched_configurations_partitioned(activation_dict; kwargs...)

Main entry point for partitioned processing. Automatically partitions the
activation_dict by value lengths and processes each partition with optimal
batch sizes while sharing a single CountMinSketch.

Each partition is processed completely (count → select → extract) before moving
to the next partition, ensuring memory efficiency and correctness.

# Arguments
- `activation_dict`: Dictionary mapping data point IDs to feature vectors
- `motif_size`: Number of features per motif (default: 3)
- `partition_width`: Width of length ranges for partitioning (default: 10)
- `filter_len`: Filter length for convolution case
- `min_count`: Minimum count threshold (default: 1, recommended for partitioned processing)
- `seed`: Random seed for reproducibility
- `config`: HyperSketchConfig object (optional)
- `verbose`: Whether to print detailed progress information (default: false)
- Other arguments passed to partition creation

# Returns
A DataFrame with enriched configurations from all partitions.

# Example
```julia
motifs = obtain_enriched_configurations_partitioned(
    activation_dict,
    motif_size=3,
    partition_width=10,
    batch_size=:auto,
    min_count=1  # Recommended: use 1, filter DataFrame afterwards
)
filtered = filter(row -> row.count >= 5, motifs)
```
"""
function obtain_enriched_configurations_partitioned(
    activation_dict::ActivationDict;
    motif_size::Integer=3,
    partition_width::Integer=10,
    filter_len::Union{Integer,Nothing}=nothing,
    min_count::Integer=1,
    seed::Union{Integer, Nothing}=1,
    batch_size::Union{Integer, Symbol}=:auto,
    auto_batch_verbose::Bool=false,
    verbose::Bool=false,
    config::Union{HyperSketchConfig, Nothing}=nothing
)
    # Create config if not provided
    if config === nothing
        config = default_config(min_count=min_count, seed=seed)
    end
    
    # Warn about min_count > 1 with partitioning
    if config.min_count > 1
        @warn """
        Using min_count=$(config.min_count) with partitioned processing.
        
        Note: Selection happens per-partition, so a motif appearing across multiple
        partitions may not be selected if it doesn't meet min_count in each individual
        partition. 
        
        Recommended approach:
        1. Set min_count=1 here to extract all motifs
        2. Filter the resulting DataFrame afterwards:
           motifs = obtain_enriched_configurations_partitioned(..., min_count=1)
           filtered = filter(row -> row.count >= desired_min_count, motifs)
        
        This ensures motifs spanning multiple partitions are correctly counted.
        """ maxlog=1
    end
    
    # Validation
    validate_activation_dict(activation_dict)
    validate_motif_size(motif_size)
    check_cuda_requirements(config.use_cuda)
    
    # Create partitioned record
    verbose && @info "Creating partitioned record..."
    pr = create_partitioned_record(        
        activation_dict, motif_size;
        partition_width=partition_width,
        batch_size=batch_size,
        use_cuda=config.use_cuda,
        filter_len=filter_len,
        seed=config.seed,
        auto_batch_verbose=auto_batch_verbose,
        verbose=verbose
    )
    
    # Execute pipeline - CRITICAL: Process each partition completely before moving to next
    # This ensures selectedCombs state is preserved within each Record
    verbose && @info "Processing partitions sequentially (count → select → extract per partition)..."
    
    dfs = Vector{DataFrame}(undef, length(pr.partitions))
    
    for (i, (partition, range)) in enumerate(zip(pr.partitions, pr.partition_ranges))
        verbose && @info "Processing partition $i/$(length(pr.partitions)) (length range: $range, $(length(partition)) data points)"
        
        # Create Record for this partition
        record = _create_record_for_partition(partition, pr, i, verbose)
        
        # Count, select, and extract in sequence (same Record!)
        verbose && @info "  Counting..."
        count!(record, config)
        
        verbose && @info "  Selecting..."
        make_selection!(record, config)
        
        verbose && @info "  Extracting..."
        dfs[i] = _obtain_enriched_configurations_(record, config)
        
        verbose && @info "  Extracted $(nrow(dfs[i])) configurations, freeing memory"
    end
    
    # Combine all DataFrames
    verbose && @info "Combining results from all partitions..."
    motifs = vcat(dfs...)
    
    verbose && @info "Extracted $(nrow(motifs)) enriched configurations total"
    
    return motifs
end

# Additional helper: print partition statistics
"""
    print_partition_stats(pr::PartitionedRecord)

Print statistics about the partitions in a PartitionedRecord.
"""
function print_partition_stats(pr::PartitionedRecord)
    println("\n" * "="^60)
    println("Partitioned Record Statistics")
    println("="^60)
    println("Number of partitions: $(length(pr.partitions))")
    println("Motif size: $(pr.motif_size)")
    println("Case: $(pr.case)")
    println("Filter length: $(pr.filter_len)")
    println("Using CUDA: $(pr.use_cuda)")
    println("Batch size: $(pr.batch_size)")
    println("\nShared CountMinSketch:")
    println("  Rows: $(size(pr.shared_cms.sketch, 1))")
    println("  Columns: $(size(pr.shared_cms.sketch, 2))")
    println("\nPartition Details:")
    println("-"^60)
    
    for (i, (partition, range)) in enumerate(zip(pr.partitions, pr.partition_ranges))
        num_data_points = length(partition)
        
        println("Partition $i (length range: $range)")
        println("  Data points: $num_data_points")
        
        if i < length(pr.partitions)
            println()
        end
    end
    
    println("="^60)
    println("\nNote: Records are created on-demand to minimize memory usage.")
end
