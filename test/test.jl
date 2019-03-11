push!(LOAD_PATH, pwd() * "/../src")

using Layers, Activations, Evaluations

model = Layers.initInput(10)
model = Layers.initDense(model, 5, Layers.sigmoid)
model = Layers.initDense(model, 5, Layers.softmax)
Evaluations.predict!(model, ones(10))

println(model.output)
println(sum(model.output))