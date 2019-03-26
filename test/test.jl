push!(LOAD_PATH, pwd() * "/../src")

using Layers, Evaluations, Losses

model = initInput(10)
model = initDense(model, 5, "sigmoid")
model = initDense(model, 5, "softmax")
predict!(model, ones(10))

println(model.ȳ)
println(sum(model.ȳ))
println(evaluate!(model, ones(10), ones(5), mse))