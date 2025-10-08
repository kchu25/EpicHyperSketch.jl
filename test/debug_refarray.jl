using EpicHyperSketch

# Check the refArray dimensions
activation_dict = Dict{Int, Vector{EpicHyperSketch.ConvolutionFeature}}()

# Insert motif [22,8,39]
motif_features = [(22, 8), (8, 20), (39, 35)]
conv_features = [(filter=f, contribution=1.0f0, position=p) for (f, p) in motif_features]
activation_dict[1] = conv_features

max_active_len = EpicHyperSketch.get_max_active_len(activation_dict)
println("max_active_len: ", max_active_len)

vecRefArray, _ = EpicHyperSketch.constructVecRefArrays(
    activation_dict, max_active_len; 
    batch_size=500, case=:Convolution, use_cuda=false
)

ref_array = vecRefArray[1]
println("refArray dimensions: ", size(ref_array))

println("\nrefArray content for sequence 1:")
for i in 1:max_active_len
    row = ref_array[i, :, 1]
    println("  Index $i: ", row)
end

# Check constants
println("\nConstants:")
println("FILTER_INDEX_COLUMN = ", EpicHyperSketch.FILTER_INDEX_COLUMN)
println("DATA_PT_INDEX_COLUMN = ", EpicHyperSketch.DATA_PT_INDEX_COLUMN)  
println("POSITION_COLUMN = ", EpicHyperSketch.POSITION_COLUMN)
