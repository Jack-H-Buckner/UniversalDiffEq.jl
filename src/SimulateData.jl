
"""
    LotkaVolterra(;kwargs)
    
    Create a sample dataset using the Lotka-Volterra predator-prey model as its process model:

    ```math
    \frac{dN}{dt} = rN - \alpha NP \\
    \frac{dP}{dt} = \theta\alpha NP - mP
    ```

    and an observation error following a normal distribution with mean 0 and standard deviation σ.

    # kwargs
    - `plot`: Does the function return a plot? Default is `true`.
    - `seed`: Seed for observation error to create repeatable examples. Default is `123`.
    - `datasize`: Number of time steps generated. Default is `60`.
    - `T`: Maximum timespan. Default is `3.0`.
    - `sigma`: Standard deviation of observation error. Default is `0.075`.
"""
function LotkaVolterra(;plot = true, seed = 123,datasize = 60,T = 3.0,sigma = 0.075)
    # set seed 
    Random.seed!(seed)

    # set parameters for data set 
    tspan = (0.0f0, T)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    # model parameters
    u0 = Float32[1.0, 0.5]
    p=(r = 4.0, alpha = 5.0, theta = 0.75, m = 5.0)
    
    # define derivatives 
    function lotka_volterra(du, u, p, t)
        du[1] = p.r*u[1] - p.alpha *u[1]*u[2]
        du[2] = p.theta*p.alpha *u[1]*u[2] - p.m*u[2]
    end

    # generate time series with DifferentialEquations.jl
    prob_trueode = ODEProblem(lotka_volterra, u0, tspan, p)
    ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

    # add observation noise 
    ode_data .+= ode_data .* rand(Normal(0.0,sigma), size(ode_data))
        
    if plot
        plt = Plots.scatter(tsteps,transpose(ode_data), xlabel = "Time", ylabel = "Abundance", label = "")
        data = DataFrame(time = tsteps, x1 = ode_data[1,:], x2 = ode_data[2,:])
        return data, plt
    end
    
    data = DataFrame(time = tsteps, x1 = ode_data[1,:], x2 = ode_data[2,:])
    
    return data
end 


"""
    LorenzLotkaVolterra(;kwargs)
    
    Create a sample dataset using the Lorenz Lotka-Volterra model as its process model:
...
    ```math
    \frac{dx}{dt} = rx(1-\frac{x}{K}) - \alpha xy + gz\\
    \frac{dy}{dt} = \theta\alpha xy - my\\
    \frac{dz}{dt} = l(w-z)\\
    \frac{dw}{dt} = z(\rho-s) 0 w\\
    \frac{ds}{dt} = zw-\beta s
    ```
...
    and an observation error following a normal distribution with mean 0 and standard deviation σ_{obs}.
...
    # kwargs
    - `plot`: Does the function return a plot? Default is `true`.
    - `seed`: Seed for observation error to create repeatable examples. Default is `123`.
    - `datasize`: Number of time steps generated. Default is `60`.
    - `T`: Maximum timespan. Default is `3.0`.
    - `sigma`: Standard deviation of observation error. Default is `0.075`.
...
"""
function LorenzLotkaVolterra(;plot = true, seed = 123,datasize = 60,T = 3.0,sigma = 0.075)
    # set seed 
    Random.seed!(seed)

    # set parameters for data set 
    tspan = (0.0f0, T)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    # model parameters
    u0 = Float32[1.0, 0.5,-10, -5.0, 20.0,]
    p=(r = 4.0, K = 10, alpha = 5.0, theta = 0.75, m = 5.0, sigma = 10, rho = 28, beta = 8/3)
    
    # define derivatives 
    function lorenz_lotka_volterra(du, u, p, t)
        du[1] = p.r*u[1]*(1-u[1]/p.K) - p.alpha *u[1]*u[2]+0.05*u[3]
        du[2] = p.theta*p.alpha *u[1]*u[2] - p.m*u[2]
        du[3] = p.sigma*(u[4]-u[3])
        du[4] = u[3]*(p.rho-u[5]) - u[3]
        du[5] = u[3]*u[4] - p.beta*u[5]
    end

    # generate time series with DifferentialEquations.jl
    prob_trueode = ODEProblem(lorenz_lotka_volterra, u0, tspan, p)
    ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

    # add observation noise 
    ode_data .+= ode_data .* rand(Normal(0.0,sigma), size(ode_data))
        
    if plot
        plt = Plots.scatter(tsteps,transpose(ode_data[1:2,:]), xlabel = "Time", ylabel = "Abundance", label = "")
        data = DataFrame(time = tsteps, x1 = ode_data[1,:], x2 = ode_data[2,:])
        X = DataFrame(time = tsteps, X = ode_data[3,:])
        return data, X, plt
    end
    
    data = DataFrame(time = tsteps, x1 = ode_data[1,:], x2 = ode_data[2,:])
    X = DataFrame(time = tsteps, X = ode_data[3,:])
    return data, X
end 

function LogisticLorenz(;plot = true, seed = 123, datasize = 100,T = 30.0,sigma = 0.0125,p = (r=0.5,sigma = 10, rho = 28, beta = 8/3))
    
    # set seed 
    Random.seed!(seed)

    # set parameters for data set 
    tspan = (0.0f0, T)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    # model parameters
    u0 = Float32[-10, -5.0, 30.0, 1.0]
    

    function Lorenz(du, u, p, t)
        du[1] = p.sigma*(u[2]-u[1])
        du[2] = u[1]*(p.rho-u[3]) - u[1]
        du[3] = u[1]*u[2] - p.beta*u[3]
        du[4] = p.r*u[4]*(1-u[4]) + 0.1*u[4]*u[1]
    end

    # generate time series with DifferentialEquations.jl
    prob_trueode = ODEProblem(Lorenz, u0, tspan, p)
    ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

    # add observation noise 
    ode_data .+= ode_data .* rand(Normal(0.0,sigma), size(ode_data))
    
    if plot
        data = DataFrame(time = tsteps, x1 = ode_data[4,:])
        plt = Plots.plot(data.time,data.x1, xlabel = "Time", ylabel = "Abundance")
        return data, plt
    end
    
    data = DataFrame(time = tsteps, x1 = ode_data[4,:])
    return data
    
end 


function LogisticMap(;plot = true, seed=123,datasize = 100,sigma = 0.05, r = 3.7)
    
    # set seed 
    Random.seed!(seed)
    
    # logistic map
    map(x) = r*x*(1.0-x)
    T = datasize
    x = 0.001
    xls = []
    for i in 1:T
        push!(xls,x)
        x = map(x).+sigma*x*(rand()-0.5)
    end
    
    if plot
        data = DataFrame(time = collect(1:T), x = xls .+ rand(Distributions.Normal(0,sigma), length(xls)))
        plt = Plots.plot(data.t,data.x, xlabel = "Time", ylabel = "Abundance")
        return data, plt
    end 
    data = DataFrame(time = collect(1:T), x = xls .+ rand(Distributions.Normal(0,sigma), length(xls)))
    return data
    
end 

