using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets
using LinearAlgebra
using IterTools
using Base.Iterators
using Plots

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
lambda = .1
eq  = Dt(u(t, x)) ~ lambda*Dxx(u(t, x))
bcs = [u(0, x) ~ 0,
        u(t, 0) ~ 1,
        u(t, 1) ~ 1]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]

println("here!")
# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

# Method of lines discretization
dx = 0.025
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob, Tsit5(), saveat=0.05)
println("here!")

# Plot results and compare with exact solution
discrete_x = sol[x]
discrete_t = sol[t]
solu = sol[u(t, x)]
ddt = stack(sol.original_sol(discrete_t, Val{1})) # cheating!
ddt = [zeros(1,size(ddt,2)); ddt; zeros(1,size(ddt,2))]

#=
using Plots
plt = plot()

for i in eachindex(discrete_t)
    plot!(discrete_x, solu[i, :], label="Numerical, t=$(discrete_t[i])")
    scatter!(discrete_x, u_exact(discrete_x, discrete_t[i]), label="Exact, t=$(discrete_t[i])")
end
display(plt)
println("here!")
=#

data = solu'

#data = (data[:,1:end-1] + data[:,2:end])/2
U, S, V = svd(data)
kron2(u) = kron(u,u)
#lift(u) = kron2([u;1])
lift(u) = [u;1]

function integrate_operator(op, initial, discrete_t)
    prob = ODEProblem((u,p,t) -> op*lift(u), initial, (0,1))
    sol = solve(prob, Tsit5(), saveat=discrete_t)
    return stack(sol(discrete_t))
end

function plot_numerical(t, x, solu)
    plt = plot()
    for i in eachindex(t)
        plot!(x, solu[:, i], label="Numerical, t=$(t[i])")
    end
    display(plt)
end

function tikhonov(A, b)
    lambda = .01
    return inv(A' * A + lambda*I(size(A,2))) * A' * b
end

function compute_errors(r::Number)
    Ur = U[:,1:r]

    # Numerical derivative

    projddt = Ur' * ddt
    projdata = Ur' * data

    display("projddt")
    display(projddt)

    # https://discourse.julialang.org/t/how-to-kronecker-by-row-one-dimension-only/33942

    lifteddata = stack(lift.(eachcol(projdata)))
    display("lifteddata")
    display(lifteddata)

    op = tikhonov(lifteddata', projddt')'
    display("op")
    display(op)

    states = integrate_operator(op, projdata[:,1], discrete_t)
    display("states");
    display(states)

    plot_numerical(discrete_t[1:10], discrete_x, Ur*states)
    relstate_err = norm(Ur*states-data)
    proj_err = norm(Ur * op * lifteddata - ddt)
    return relstate_err, proj_err
end

plt = plot()
rs = 1:10
errors = [compute_errors(r) for r in rs]
plot!(rs, getindex.(errors,1), label="Relative State Error")
plot!(rs, getindex.(errors,2), label="Projection Error")
