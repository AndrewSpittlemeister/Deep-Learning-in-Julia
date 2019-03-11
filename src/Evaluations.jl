module Evaluations
export predict!

using Layers, Activations

function predict!(layer::Layers.Layer, input::Vector{Float64})
    if typeof(layer) == Layers.Input
        if size(input)[1] != layer.size
            error("Input dimensions were $(size(input)[1]) but should be $(layer.size).")
        else
            layer.output = input
        end
    else
        predict!(layer.previous, input)
        Activations.activate!(layer)
    end
end

end
