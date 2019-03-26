module Optimizations
export train!, fit!

using Layers, Activations, Losses

function train!(model::Layers.Layer, X::Matrix{Float64}, ȳ::Vector{Float64}, loss::Function, η::Float64)

end

function fit!(model::layers.Layer, X::Matrix{Float64}, ȳ::Vector{Float64}, loss::Function, η::Float64, epochs::Int64)   

end

end