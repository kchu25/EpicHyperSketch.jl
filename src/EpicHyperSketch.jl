module EpicHyperSketch

using CUDA
using Combinatorics
using Random
using DataFrames
using ProgressMeter
#

const IntType = Int32 # RTX 3090 has dedicated INT32 execution units in each SM 
const FloatType = Float32
# const PrimeNumber = IntType(50000101)  # A large prime number for hashing
# Default error probability (delta) for Count-Min Sketch.
const DEFAULT_CMS_DELTA = 0.0001  
# Default error tolerance (epsilon) for Count-Min Sketch.
const DEFAULT_CMS_EPSILON = 0.00005
# Default minimum count threshold for enriched configuration.
const default_min_count = 1
# Default batch_size for refArray in vecRefArray
const BATCH_SIZE = 500

# Constants for refArray indexing (features/filters data)
const FILTER_INDEX_COLUMN = 1  # Filter ID / Feature ID
const DATA_PT_INDEX_COLUMN = 2  # Data point index in the data set (for future use)
const POSITION_COLUMN = 3     # Position in sequence

# Number of columns (2nd dimension) in refArray based on case
const refArraysDim = Dict(
    :OrdinaryFeatures => 2,
    :Convolution => 3
)

include("types.jl")
include("errors.jl") 
include("config.jl")
include("performance.jl")
include("sketch.jl")
include("memory.jl")
include("record.jl")
include("partition.jl")
include("count_gpu.jl")
include("count_gpu_extract.jl")
include("count_cpu.jl")
include("count_cpu_extract.jl")




export CountMinSketch, 
       Record,
       PartitionedRecord,
       HyperSketchConfig,
       default_config,
       obtain_enriched_configurations,
       obtain_enriched_configurations_partitioned,
       create_partitioned_record,
       partition_by_length,
       print_partition_stats,
       # CPU versions
       obtain_enriched_configurations_cpu,
       count_cpu!,
       make_selection_cpu!,
       # Memory management
       calculate_optimal_batch_size,
       auto_configure_batch_size,
       print_memory_report,
       estimate_memory_per_batch,
       estimate_fixed_memory,
       # Errors
       HyperSketchError,
       InvalidConfigurationError,
       CUDAError

end
