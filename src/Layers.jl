module Layers
export initInput, initDense, Layer, Dense, Input, sigmoid, relu, linear

@enum ActivationFunction softmax sigmoid relu linear

abstract type Layer end

mutable struct Dense <: Layer
    previous::Layer
    size::Int64
    weights::Matrix{Float64}
    output::Vector{Float64}
    activation::ActivationFunction

end

function initDense(previous::Layer, size::Int64, activation::ActivationFunction)::Dense
    return Dense(previous, size, rand(size, previous.size), zeros(size), activation)
end

mutable struct Input <: Layer
    size::Int64
    output::Vector{Float64}
end

function initInput(size::Int64)::Input
    return Input(size, ones(size))
end

end