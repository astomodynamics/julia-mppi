#=
###############################################################################
#    Model Predictive Path Integral Control Algorithm is implemented
###############################################################################
Refs:
    [1] Williams, G., Aldrich, A., and Theodorou, E. A.
    “Model Predictive Path Integral Control: From Theory to Parallel Computation.”
    Journal of Guidance, Control, and Dynamics, Vol. 40, No. 2, 2017, pp. 1–14.
    https://doi.org/10.2514/1.g001921.
=#

# using CUDA
# Dynamics model
abstract type AbstractModel end

"""
    solve_mppi(model, mppi_problem, X0, U0)

Solves the MPPI control problem and returns solution trajectories of state and
control inputs.

# Arguments
- `model`: Abstract model
- `problem`: MPPI problem definitions

# Return
- `X`: state trajectory
- `U`: control trajectory

# Examples
```julia

model = DynamicalModel()
mppi_problem = MPPIProblem(model, model.dt, model.tN, model.hN)
X, U = solve_mppi_cpu(model, problem)

```
"""
function solve_mppi_cpu(model::AbstractModel, problem::MPPIProblem)
    # setup simulation parameters
    dt, tN, hN = model.dt, model.tN, model.hN
    x_dim, u_dim = model.x_dim, model.u_dim
    K = problem.K # number of samples
    x_init = model.x_init # initial state, but updated in the horizon loop
    d = model.distribution # normal distribution with zero mean and variance
    # ρ = problem.ρ
 
    # initialization for trajectories
    X = zeros(x_dim, tN)
    U = zeros(u_dim, tN)
    Xh= zeros(x_dim, hN)
    Uh = zeros(u_dim, hN)
    δu = zeros(u_dim, hN, K)
    X[:,1] = model.x_init

    # main loop
    for t in 1:tN-1
        # Initializaiton of cost for K samples
        S = zeros(K)

        # computing cost for K samples and N finite horizon
        for k in 1:K
            Xh[:,1] = x_init

            for j in 1:hN-1
                δu[:, j, k] = rand(d, u_dim)
                Xh[:, j+1] = Xh[:, j] + rk4_step(model, model.dynamics, j, Xh[:, j], Uh[:, j] + δu[:, j, k], dt) * dt
                S[k] += state_cost(model, Xh[:,j]) + get_sampling_cost_update(model, problem, Xh[:,j], Uh[:,j], δu[:, j, k])
                # S[k] += state_cost(model, Xh[:,j]) * dt
            end
            # S[k] += terminal_cost(model, Xh[:, hN])
            δu[:, hN, k] = rand(d, u_dim)
        end

        # compute control for horizon 
        for j in 1:hN
            Uh[:, j] += get_optimal_distribution(model, problem, S, δu[:, j, :])
        end
        
        # export to system actuators
        U[:, t] = Uh[:, 1]

        # simulate the system with the updated input
        X[:, t+1] = X[:, t] + rk4_step(model, model.dynamics, t, X[:, t], U[:, t], dt) * dt
        
        # update control sequence for the next time step
        for j in 1:hN-1
            Uh[:, j] = Uh[:, j+1]
        end

        # add last control as randomized input
        Uh[:, end] = rand(d, u_dim)

        # update the initial state for the next horizon
        x_init = X[:, t+1]
    end

    return X, U
end

"""
    get_total_entropy(model, problem, S, δu)
 
Compute total entropy at a specific time

"""
function get_optimal_distribution(
    model::AbstractModel,
    problem::MPPIProblem,
    S::AbstractArray{Float64,1},
    δu::AbstractArray{Float64, 2},
)
    # normalization of cost function if needed
    # S = S/sum(S)
    K = length(S) # number of samples
    λ = problem.λ # temperature
    sum1 = zeros(model.u_dim)
    sum2 = 0
    for k in 1:K
        sum1 += exp(-(1 / λ) * S[k]) * δu[:, k]
        sum2 += exp(-(1 / λ) * S[k])
    end 
    return sum1/sum2
end