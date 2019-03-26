module Losses
export mse, crossentropy

function mse(ȳ::Vector{Float64}, ŷ::Vector{Float64})::Float64
    return sum((ŷ - ȳ).^2) / length(ȳ)
end

function crossentropy(ȳ::Vector{Float64}, ŷ::Vector{Float64})::Float64
    return -sum(ȳ .* map(log, ŷ))
end

end
