module Layers
export initInput, initDense, Layer

abstract type Layer end

mutable struct Dense <: Layer
    previous::Layer
    size::Int64
    W::Matrix{Float64}
    z̄::Vector{Float64}
    ȳ::Vector{Float64}
    activation::String
end

function initDense(previous::Layer, size::Int64, activation::String)::Dense
    return Dense(previous, size, rand(size, previous.size), zeros(size), zeros(size), activation)
end

mutable struct Input <: Layer
    size::Int64
    ȳ::Vector{Float64}
end

function initInput(size::Int64)::Input
    return Input(size, ones(size))
end

end