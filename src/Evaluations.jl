module Evaluations
export predict!, evaluate!

using Layers, Activations, Losses

function predict!(layer::Layers.Layer, input::Vector{Float64})
    if typeof(layer) == Layers.Input
        if size(input)[1] != layer.size
            error("Input dimensions were $(size(input)[1]) but should be $(layer.size).")
        else
            layer.ȳ = input
        end
    else
        predict!(layer.previous, input)
        activate!(layer)
    end
end

function evaluate!(model::Layers.Layer, input::Vector{Float64}, output::Vector{Float64}, loss::Function)::Float64
    predict!(model, input)
    return loss(output, model.ȳ)
end

end
