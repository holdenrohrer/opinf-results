function get_heat_equation_data()
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    mu = 10
    eq  = Dt(u(t, x)) ~ mu*Dxx(u(t, x))
    bcs = [u(0, x) ~ 0,
            u(t, 0) ~ 1,
            u(t, 1) ~ 1]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
               x ∈ Interval(0.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2^(-7)
    discretization = MOLFiniteDifference([x => dx], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)

    # Solve ODE problem
    sol = solve(prob, ImplicitEuler(), saveat=10^(-3))

    # Plot results and compare with exact solution
    discrete_x = sol[x]
    discrete_t = sol[t]
    solu = sol[u(t, x)]
    ddt = stack(sol.original_sol(discrete_t, Val{1})) # cheating!
    ddt = [zeros(1,size(ddt,2)); ddt; zeros(1,size(ddt,2))]

    return discrete_x, discrete_t, ddt, solu

    data = solu'
end

function integrate_operator(op, initial, discrete_t)
    prob = ODEProblem((u,p,t) -> op*lift(u), initial, (0,1))
    sol = solve(prob, ImplicitEuler(), saveat=discrete_t)
    return stack(sol(discrete_t))
end

function plot_numerical(t, x, solu)
    display(surface(t, x, solu))
end

function tikhonov(A, b)
    lambda = 0.01
    return inv(A' * A + lambda*I(size(A,2))) * A' * b
end

function compute_projection_and_errors(r::Number, discrete_x::StepRangeLen{Float64}, discrete_t::Vector{Float64}, ddt::Matrix{Float64}, solu::Matrix{Float64})
    U, S, V = svd(data)
    lift(u) = [u;1]
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

    #plot_numerical(discrete_t, discrete_x, Ur * projdata)
    relstate_err = norm(Ur*states - data)/norm(data)
    proj_err = norm(Ur * projdata - data)/norm(data)
    return relstate_err, proj_err
end
