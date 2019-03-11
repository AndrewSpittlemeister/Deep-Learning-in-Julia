module Activations
export activate!, softmax!, sigmoid!, relu!, linear!

using Layers 

function activate!(layer::Layers.Layer)
    if typeof(layer) == Layers.Input
        return
    else
        if layer.activation == Layers.softmax
            softmax!(layer)
        elseif layer.activation == Layers.sigmoid
            sigmoid!(layer)
        elseif layer.activation == Layers.relu
            relu!(layer)
        elseif layer.activation == Layers.linear
            linear!(layer)
        else
            throw(DomainError(layer.activation, "invalid activation type"))
        end
    end
end

function softmax!(layer::Layers.Layer)
    if typeof(layer) == Layers.Dense
        sigmoid!(layer)
        layer.output ./= sum(layer.output)
    end
end

function sigmoid!(layer::Layers.Layer)
    linear!(layer)
    if typeof(layer) == Layers.Dense
        map(x -> 1 / (1 + ℯ^(-x)), layer.output)
    end
end

function relu!(layer::Layers.Layer)
    linear!(layer)
    if typeof(layer) == Layers.Dense
        map(x -> max(0,x), layer.output)
    end
end

function linear!(layer::Layers.Layer)
    layer.output = layer.weights * layer.previous.output
end

end