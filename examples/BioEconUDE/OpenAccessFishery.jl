module OpenAccessFishery

using Plots, DifferentialEquations, Distributions, DataFrames

mutable struct parameters
    u0::AbstractVector{Float64}
    growth_rate::Float64 
    carrying_capacity::Float64
    search_efficancy_0::Float64
    tech_imporvement::Float64
    handeling_time::Float64
    k::Float64
    depreciation::Float64
    entry::Float64
    exit::Float64
    threshold_CPUE::Float64
end 



function derivs(u,p,t)
    X = u[1]
    K = u[2]
    search_efficancy = p.search_efficancy_0 * exp(p.tech_imporvement * t)
    CPUE = search_efficancy*X^p.k/(1 + p.handeling_time*search_efficancy*X^p.k)
    dX = p.growth_rate*X*(1-X/p.carrying_capacity) - CPUE * K
    dK = -1*p.depreciation*K
    dH = CPUE * K
    if CPUE > p.threshold_CPUE
        dK += p.entry*(CPUE-p.threshold_CPUE)*K
    else
        dK += p.exit*(CPUE-p.threshold_CPUE)*K
    end
    return [dX,dK,dH]
end 


    
function noise(u, p, t)
    dW = zeros(3)
    dW[1] = u[1]*p.sigma_X
    dW[2] = u[2]*p.sigma_K
    return dW
end


function simulation(p,tspan,nsteps,dt)
    tsteps = collect(tspan[1]:((tspan[2]-tspan[1])/nsteps):tspan[2])
    prob = SDEProblem(derivs,noise, p.u0, tspan, p)
    ode_data = solve(prob, SRIW1(), dt = dt, adaptive = false, saveat = tsteps)
end 



function simulate_data(p;sigma = [0.1,0.1,0.1],variables = ["Capital","Stock","Production"], tspan = [0.0,50.0],nsteps = 100,dt = 0.1)

    sol = simulation(p,tspan,nsteps,dt)
 
    t = sol.t
    X = broadcast(i -> sol.u[i][1], 1:length(t))   
    
    K = broadcast(i -> sol.u[i][2], 1:length(t))
        
    H = broadcast(i -> sol.u[i][3], 1:length(t))
       
    data = DataFrame()
    data.t = t[1:(end-1)]
    
    if "Production" in variables
        Catch = H[2:end] .- H[1:(end-1)] 
        Catch .+= log.(Catch) .+ sigma[1] .* rand(Distributions.Normal(),length(Catch))
        data.Production = Catch
    end
    
    if "Capital" in variables
        data.Capital = log.(K[1:(end-1)]) .+  sigma[2] .* rand(Distributions.Normal(),length(K[2:end]))
            
    end
    
    if "Stock" in variables
        
        data.Stock = log.(X[1:(end-1)]) .+ sigma[3] .* rand(Distributions.Normal(),length(K[2:end]))
            
    end 
     
    return data
end 


function sim_true(p,tsteps;variables = ["Production"])
    prob = ODEProblem(derivs!, p.u0, [0.0,tsteps[argmax(tsteps)]])

    sol = solve(prob, Tsit5(), p = p, saveat = tsteps)
    t = sol.t
    X = broadcast(i -> sol.u[i][1], 1:length(t))   
    
    K = broadcast(i -> sol.u[i][2], 1:length(t))
        
    H = broadcast(i -> sol.u[i][3], 1:length(t))
       
    data = DataFrame()
    data.time = t
    
    data.Capital = log.(K)      

    data.Stock = log.(X)       
     
    return data
end 




function makeplot(p;tspan = [0.0,100.0],nsteps = 100)
    sol = OpenAccessFishery.simulation(p;tspan = tspan,nsteps =nsteps)
    t = sol.t
    X = broadcast(i -> sol.u[i][1], 1:length(t))
    K = broadcast(i -> sol.u[i][2], 1:length(t))
    H = broadcast(i -> sol.u[i][3], 1:length(t))
    p1=plot(t,X, color = "black", label = "Resource", ylabel = "Abundance")
    plot!(t,K, color = "black", linestyle = :dash, label = "Captial" )
    p2=plot(X,K, color = "black", xlabel = "Resource", ylabel = "Capital", label = false)
    p3=plot(t[2:end],H[2:end].-H[1:(end-1)], color = "black", xlabel = "time", ylabel = "Production", label = false)
    plot(p1,p2,p3)
end 


function plot_simulation(data)
    plt = plot(xlabel = "time", ylabel = "Abundance")
    p1 = plot()
    if "Production" in names(data)
       Plots.scatter!(plt,data.t,data.Production, color = "black", label = "Production")
        Plots.scatter!(p1,data.t,data.Production, color = "black", label = "Production")
    end
    p2 = plot()
    if "Capital" in names(data)
       Plots.scatter!(plt,data.t,data.Capital, color = "white",label = "Capital")
        Plots.scatter!(p2,data.t,data.Capital, color = "white", label = "Capital")
    end
    p3 = plot()
    if "Stock" in names(data)
       Plots.scatter!(plt,data.t,data.Stock, color = "grey",label = "Stock")
        Plots.scatter!(p3,data.t,data.Stock, color = "grey", label = "Stock")
    end

    return plt,p1,p2,p3
end 

stable_eq = (u0 = [10.0,0.05,0.0],growth_rate = 0.3,carrying_capacity = 10.0,search_efficancy_0 = 0.11,
            tech_imporvement = 0.0,handeling_time = 0.0,k= 1.25,depreciation = 0.0,entry = 0.15,exit = 0.15,
            threshold_CPUE = 0.5, sigma_X = 0.00,sigma_K = 0.0)

damped_osc = (u0 = [10.0,0.05,0.0],growth_rate = 0.3,carrying_capacity = 10.0,search_efficancy_0 = 0.25,
            tech_imporvement = 0.0,handeling_time = 0.0,k= 1.0,depreciation = 0.0,entry = 0.15,exit = 0.15,
            threshold_CPUE = 0.5, sigma_X = 0.00,sigma_K = 0.0)

limit_cyc = (u0 = [10.0,0.05,0.0],growth_rate = 0.3,carrying_capacity = 10.0,search_efficancy_0 = 0.65,
            tech_imporvement = 0.0,handeling_time = 0.3,k= 1.1,depreciation = 0.0,entry = 0.15,exit = 0.15,
            threshold_CPUE = 0.5, sigma_X = 0.0,sigma_K = 0.0)

end # module