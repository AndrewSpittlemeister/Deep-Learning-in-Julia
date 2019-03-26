module Activations
export activate! #, activate′!

using Layers, Util

# TODO: finish off activation derivatives

function activate!(layer::Layers.Layer)
    if typeof(layer) == Layers.Input
        return
    else
        ActivationLookup[layer.activation](layer)
    end
end

# function activate′!(layer::Layers.Layer)::Matrix{Float64}
#     if typeof(layer) == Layers.Input
#         return
#     else
#         DerivativeLookup[layer.activation](layer)
#     end
# end

function softmax!(layer::Layers.Layer)
    if typeof(layer) == Layers.Dense
        sigmoid!(layer)
        layer.ȳ ./= sum(layer.z̄)
    end
end

# function softmax′!(Layer::Layers.Layer)::Matrix{Float64}
#     
# end

function sigmoid!(layer::Layers.Layer)
    linear!(layer)
    if typeof(layer) == Layers.Dense
        map(x -> 1 / (1 + ℯ^(-x)), layer.z̄)
    end
end

# function sigmoid′!(layer::Layers.Layer)::Matrix{Float64}
#
# end

function relu!(layer::Layers.Layer)
    linear!(layer)
    if typeof(layer) == Layers.Dense
        map(x -> max(0,x), layer.z̄)
    end
end

# function relu′!(layer::Layers.Layer)::Matrix{Float64}
#
# end

function linear!(layer::Layers.Layer)
    layer.z̄ = layer.W * layer.previous.ȳ
    layer.ȳ = copy(layer.z̄)
end

# function linear′!(layer::Layers.Layer)::Matrix{Float64}
#     return ones(layer.size, layer.size)
# end

ActivationLookup = Dict{String, Function}(
    "softmax" => softmax!,
    "sigmoid" => sigmoid!,
    "relu" => relu!,
    "linear" => linear!
)

# DerivativeLookup = Dict{String, Function}(
#     "softmax" => softmax′!,
#     "sigmoid" => sigmoid′!,
#     "relu" => relu′!,
#     "linear" => linear′!
# )

end