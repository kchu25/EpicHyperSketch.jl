# Simple Example: Finding Enriched Motifs

using EpicHyperSketch
using Random

println("="^60)
println("EpicHyperSketch: Simple Example")
println("="^60)

# Create example data: sequences with feature activations
Random.seed!(42)

# Simulate 100 sequences with 5-15 features each
# Features are integers from 1-20
activation_dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int32, Float32}}}}()

for i in 1:100
    num_features = rand(5:15)
    features = [(feature=Int32(rand(1:20)), contribution=Float32(rand())) 
                for _ in 1:num_features]
    activation_dict[i] = features
end

println("\nCreated $(length(activation_dict)) sequences")
println("Features per sequence: $(minimum(length(v) for v in values(activation_dict)))-$(maximum(length(v) for v in values(activation_dict)))")

# Find 2-feature motifs
println("\n" * "-"^60)
println("Finding 2-feature motifs (min_count=5)...")
println("-"^60)

motifs = obtain_enriched_configurations(
    activation_dict;
    motif_size=2,
    batch_size=:auto,
    min_count=5
)

println("Found $(nrow(motifs)) motif occurrences")

if nrow(motifs) > 0
    # Show unique motifs with counts
    using DataFrames
    unique_motifs = combine(
        groupby(motifs, [:m1, :m2]), 
        :count => first => :total_count
    )
    sort!(unique_motifs, :total_count, rev=true)
    
    println("\nTop 10 most frequent motifs:")
    println(first(unique_motifs, min(10, nrow(unique_motifs))))
else
    println("No motifs found with min_count=5")
    println("Try lowering min_count or generating more sequences")
end

# Example with partitioned processing (for variable-length sequences)
println("\n" * "="^60)
println("Example: Partitioned Processing")
println("="^60)

# Create data with highly variable lengths
variable_dict = Dict{Int, Vector{NamedTuple{(:feature, :contribution), Tuple{Int32, Float32}}}}()

# Short sequences (5-10 features)
for i in 1:30
    num_features = rand(5:10)
    features = [(feature=Int32(rand(1:15)), contribution=Float32(rand())) 
                for _ in 1:num_features]
    variable_dict[i] = features
end

# Long sequences (30-50 features)  
for i in 31:60
    num_features = rand(30:50)
    features = [(feature=Int32(rand(1:15)), contribution=Float32(rand())) 
                for _ in 1:num_features]
    variable_dict[i+30] = features
end

println("Created dataset with variable lengths:")
println("  30 short sequences (5-10 features)")
println("  30 long sequences (30-50 features)")

# Use partitioned processing
motifs_partitioned = obtain_enriched_configurations_partitioned(
    variable_dict;
    motif_size=2,
    partition_width=10,
    batch_size=:auto,
    min_count=1  # Use 1, filter afterwards
)

println("\nExtracted $(nrow(motifs_partitioned)) motif occurrences")

# Filter by count
if nrow(motifs_partitioned) > 0
    filtered = filter(row -> row.count >= 5, motifs_partitioned)
    println("After filtering (count >= 5): $(nrow(filtered)) occurrences")
    
    if nrow(filtered) > 0
        unique_filtered = combine(
            groupby(filtered, [:m1, :m2]), 
            :count => first => :total_count
        )
        sort!(unique_filtered, :total_count, rev=true)
        
        println("\nTop 5 motifs:")
        println(first(unique_filtered, min(5, nrow(unique_filtered))))
    end
end

println("\n" * "="^60)
println("Done!")
println("="^60)
