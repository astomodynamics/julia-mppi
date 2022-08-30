#=
    MPPI solver environment
=#

# Dynamics model
abstract type AbstractModel end

export MPPIProblem

# MPPI solver as a Julia class
mutable struct MPPIProblem
    dt::Any # discretization step-size
    tN::Any # number of time-discretization steps
    hN::Any # number of horizon time-discretization steps

    K::Int64 # number of samples

    # MPPI params:
    λ::Float64 # temperature in entropy
    variance::Float64 # variance of dynamics noise
    ν::Float64 # exploration parameter
    ρ::Any
    threads::Int64 # number of thread for GPU acceleration
end

# The problem is set in the class constructor
function MPPIProblem(model::AbstractModel, dt::Float64, tN::Int64, hN::Int64)
    # dt = model.dt
    # tN = model.tN
    # hN = model.hN
    
    K = 1000
    λ = 100.
    variance = model.variance
    ν = 100.
    ρ = nothing
    threads = 10
    MPPIProblem(dt, tN, hN, K, λ, variance, ν, ρ, threads)
end

"""
    get_sampling_cost_update(model, problem, x, u, δu)

update cost function 
"""
function get_sampling_cost_update(
    model::AbstractModel, 
    problem::MPPIProblem,
    x::AbstractArray{Float64,1}, 
    u::AbstractArray{Float64,1},
    δu::AbstractArray{Float64,1},
)
    ν = problem.ν
    R = model.R
    ℓ = (1 - ν^-1) / 2 * transpose(δu) * R * δu + transpose(u) * R * δu + control_cost(model,u)
    return ℓ
end

