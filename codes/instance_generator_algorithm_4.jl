#= 

This code generates an instance of (BoxQP) by using Algorithm 4 in Qiu and Yildirim (2023) and solves that instance by CPLEX.

The instance generated has an exact SDP-RLT relaxation but an inexact RLT relaxation, and a known optimal solution.

INPUT

n_1: Dimension

seed_1: Seed of the random number generator

tim_lim: Time limit for CPLEX

OUTPUT

Q, c: Parameters of (BoxQP) 

optval: Optimal value of (BoxQP)

solve_time(model_Box_QP): CPLEX solution time 

objective_value(model_Box_QP): Best objective function value found by CPLEX

objective_bound(model_Box_QP): Best lower bound found by CPLEX

relative_gap(model_Box_QP): Relative gap returned by CPLEX

termination_status(model_Box_QP): Termination status returned by CPLEX

=#


using JuMP
using LinearAlgebra
using CPLEX
using Distributions
using Random

function algorithm_four_instances(n_1,seed_1,tim_lim)

    # Initialise the seed
    
    Random.seed!(seed_1)

    # Generate L, B, and U
    
    L = []
    
    B = []
    
    U = []
    
    random_vector = rand(Uniform(0,1), n_1,1)

    for i in 1:length(random_vector)
        if random_vector[i] <= 1/3
            push!(L,i)
        elseif random_vector[i] <= 2/3
            push!(B,i)
        else 
            push!(U,i)
        end
    end
    
    num_L = length(L)
    
    num_B = length(B)
    
    num_U = length(U)

    # Ensure that B is nonempty

    if num_B <= 0
        if num_L > 0
            i = rand(1:num_L)
            push!(B,L[i])
            num_B = length(B)
            popat!(L,i)
            num_L = length(L)
        else
            i = rand(1:num_U)
            push!(B,U[i])
            num_B = length(B)
            popat!(U,i)
            num_U = length(U)
        end
    end

    # Optimal solution of BoxQP
    
    x_hat = zeros(n_1)
    
    x_hat[U] = ones(num_U)
    
    x_hat[B] = rand(0.01:0.01:0.99,num_B,1)

    # Generate r and s

    r = zeros(n_1,1)
    
    s = zeros(n_1,1)
    

    r[U] = rand(0:10, num_U, 1)
    
    s[L] = rand(0:10, num_L, 1)

    # Generate W, Y, and Z

    W = zeros(n_1,n_1)
    
    W[L,U] = rand(0:10, num_L, num_U)
    
    W[U,L] = W[L,U]'
    
    W[B,U] = rand(0:10, num_B, num_U)
    
    W[U,B] = W[B,U]'
    
    W[U,U] = Symmetric(rand(0:10,num_U,num_U))


    Y = zeros(n_1,n_1)
    
    Y[L,L] = rand(0:10, num_L, num_L)
    
    Y[B,L] = rand(0:10, num_B, num_L)
    
    Y[U,L] = rand(0:10, num_U, num_L)
    
    Y[U,B] = rand(0:10, num_U, num_B)
    
    Y[U,U] = rand(0:10, num_U, num_U)


    Z = zeros(n_1, n_1)
    
    Z[L,L] = Symmetric(rand(0:10,num_L,num_L))
    
    Z[L,B] = rand(0:10,num_L,num_B)
    
    Z[B,L] = Z[L,B]'
    
    Z[L,U] = rand(0:10, num_L, num_U)
    
    Z[U,L] = Z[L,U]'

    # Generate H, h, and beta

    A = rand(-5:5,n_1,n_1)
    
    Q1,R1 = qr(A)
    
    Omega = diagm(rand(1:10,n_1))
    
    H = Q1 * Omega * Q1'

    
    h = - H * x_hat

    
    beta = - dot(h,x_hat)
    
    # Generate Q and c

    Q = W - Y - Y' + Z + H
    
    c = -r + s - W * ones(n_1,1) + Y' * ones(n_1,1) + h

    # Optimal value of BoxQP

    optval = 0.5 * dot(Q*x_hat,x_hat) + dot(c,x_hat)
    
    # Set up CPLEX for solving BoxQP

    model_Box_QP = Model(CPLEX.Optimizer)

    set_optimizer_attribute(model_Box_QP, "CPXPARAM_OptimalityTarget", 3) # optimality target for a nonconvex problem
    
    set_optimizer_attribute(model_Box_QP, "CPXPARAM_TimeLimit", tim_lim) # time limit
    
    set_optimizer_attribute(model_Box_QP, "CPX_PARAM_THREADS", 1) # number of threads

    @variable(model_Box_QP, x[1:n_1])

    @objective(model_Box_QP, Min,  0.5*dot(Q*x,x) + dot(c,x))

    @constraint(model_Box_QP, zeros(n_1).<= x .<=ones(n_1))

    JuMP.optimize!(model_Box_QP)

    # Output results on screen

    println()

    println("Solution status: ",termination_status(model_Box_QP))

    println()

    println("Best objective function value : ",objective_value(model_Box_QP))

    println()

    println("Best feasible solution: ",value.(x))

    println()

    println("Best lower bound: ",objective_bound(model_Box_QP))

    println()

    println("Relative gap: ",relative_gap(model_Box_QP))

    println()

    println("Solution time: ",solve_time(model_Box_QP))

    println()

    println("L: ",L)

    println()

    println("B: ",B)

    println()

    println("U: ",U)
    
    println()

    println("Optimal value: ",optval)

    return [Q, c, optval,solve_time(model_Box_QP),objective_value(model_Box_QP),objective_bound(model_Box_QP),relative_gap(model_Box_QP),termination_status(model_Box_QP)] 

end
