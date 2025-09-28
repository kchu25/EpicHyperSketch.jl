module EpicHyperSketch

using CUDA
using Combinatorics
#

const IntType = Int32 # RTX 3090 has dedicated INT32 execution units in each SM 
const PrimeNumber = IntType(50000101)  # A large prime number for hashing
# Default error probability (delta) for Count-Min Sketch.
const DEFAULT_CMS_DELTA = 0.0001  
# Default error tolerance (epsilon) for Count-Min Sketch.
const DEFAULT_CMS_EPSILON = 0.00005  
# Default batch_size for refArray in vecRefArray
const BATCH_SIZE = 500

# Constants for refArray indexing (features/filters data)
const FILTER_INDEX_COLUMN = 1  # Filter ID / Feature ID
const POSITION_COLUMN = 2      # Position in sequence

# Number of columns (2nd dimension) in refArray based on case
const refArraysSecondDim = Dict(
    :OrdinaryFeatures => 1,
    :Convolution => 2
)

include("sketch.jl")
include("record.jl")



function obtain_enriched_configurations(
    selected_features;
    min_count::IntType=1,
    delta::Float64=DEFAULT_CMS_DELTA,
    epsilon::Float64=DEFAULT_CMS_EPSILON
)
    @assert min_count > 0 "min_count must be greater than 0"
    @assert 0 < delta < 1 "delta must be in (0, 1)"
    @assert 0 < epsilon < 1 "epsilon must be in (0, 1)"


    # make record
    # do the counting
    # fill in the candidates that meet the min_count threshold
    # get the enriched configurations

end


export CountMinSketch

end
