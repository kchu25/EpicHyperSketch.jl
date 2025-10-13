# Core type definitions and constants

# Type aliases for clarity
const ActivationDict{T,S} = Dict{T, Vector{S}} where {T <: Integer, S}
const RefArray3D = AbstractArray{IntType, 3}
const HashMatrix = AbstractMatrix{IntType}
const SketchMatrix = AbstractMatrix{IntType}

# Feature types
const OrdinaryFeature = NamedTuple{(:feature, :contribution), Tuple{Integer, FloatType}}
const ConvolutionFeature = NamedTuple{(:filter, :contribution, :position), Tuple{Integer, FloatType, Integer}}
# TODO: Int can be UInt; can use less space

make_conv_feature(filter::Integer, contribution::Real, position::Integer) = 
    (filter=convert(IntType, filter), contribution=contribution, position=convert(IntType, position))


# Enum for processing cases
@enum ProcessingCase OrdinaryFeatures=1 Convolution=2

# Convert between Symbol and Enum
symbol_to_case(s::Symbol) = s == :OrdinaryFeatures ? OrdinaryFeatures : Convolution
case_to_symbol(c::ProcessingCase) = c == OrdinaryFeatures ? :OrdinaryFeatures : :Convolution

# Validation functions
validate_motif_size(motif_size::Integer) = motif_size > 0 || error("motif_size must be positive")
validate_probability(p::Float64, name::String) = (0 < p < 1) || error("$name must be in (0, 1)")
validate_min_count(count::Integer) = count > 0 || error("min_count must be positive")