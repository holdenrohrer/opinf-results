using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets
using LinearAlgebra
using IterTools
using Base.Iterators
using Plots
include("heat_eq_lib.jl")

discrete_x, discrete_t, ddt, solu = get_heat_equation_data()

plt = plot()
rs = 1:7
errors = [compute_projection_and_errors(r, discrete_x, discrete_t, ddt, solu) for r in rs]
plot!(rs, getindex.(errors,1), label="Relative State Error", yaxis=:log)
plot!(rs, getindex.(errors,2), label="Projection Error", yaxis=:log)
