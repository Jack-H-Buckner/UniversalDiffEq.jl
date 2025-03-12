
function sigma_points(x̄,Px ,α,L,β,κ)
    λ = α^2*(L-κ)-L
    Wm0 = λ/(λ+L)
    Wc0 = λ/(λ+L) + (1-α^2+β)
    Wi = 1/(2*(L+λ))
    return cholesky(Hermitian((λ+L)*Px)).U, Wm0, Wc0, Wi
end

function ukf_update(y,x̄,Px,t,dt,f,p,H,Pν,Pη,L,α,β,κ)


    X, Wm0, Wc0, Wi = sigma_points(x̄,Px,α,L,β,κ)

    # calcaulte mean
    x1 = f(x̄,t,dt,p)
    xplus = [f(x̄.+X[i,:],t,dt,p) for i in 1:L]
    xminus = [f(x̄.-X[i,:],t,dt,p) for i in 1:L]
    x̄1 = Wm0*x1

    for i in 1:L
        x̄1 = x̄1 .+ Wi*xplus[i]
    end 
    for i in 1:L
        x̄1 = x̄1 .+ Wi*xminus[i]
    end 

    
    # calcualte covariance 
    Px1 = Wc0*(x1.-x̄1).*(x1.-x̄1)'
    for i in 1:L
        fx = xplus[i]
        Px1 = Px1 .+ Wi*(fx.-x̄1).*(fx.-x̄1)'
    end 
    for i in 1:L
        fx = xminus[i]
        Px1 = Px1 .+ Wi*(fx.-x̄1).*(fx.-x̄1)'
    end 
    Px1 = Px1 .+ Pν

    ŷ = H*x̄1
    S = H*Px1*H'.+Pη

    Sinv = inv(S)
    K = Px1*H'*Sinv
    x̂=x̄1.+K*(y.-ŷ)

    Px = Px1 - K*H*Px1
 
    # loglik 
    ll = -1/2 * (y.-ŷ)' * Sinv * (y.-ŷ) - 1/2 * log(det(S))  - L/2*log(2*3.14159)
    
    return x̂, Px , ll
end 



function ukf_smoothing(y,times,f,pf,H,Pν,Pη,L,α,β,κ)
    T = size(y)[2]
    xls = zeros(size(y))
    Pxls = zeros(size(y)[1],size(y)[1],size(y)[2])
    x̂ = y[:,1]
    Px = Pη
    xls[:,1] =  x̂ 
    for t in 2:T
        dt = times[t]-times[t-1]
        x̂, Px, llt = ukf_update(y[:,t],x̂,Px,times[t],dt,f,pf,H,Pν,Pη,L,α,β,κ)
        xls[:,t] = x̂
        Pxls[:,:,t] = Px 
    end
    return xls,Pxls   
end



function ukf_likelihood(y,times,f,pf,H,Pν,Pη,L,α,β,κ)
    T = size(y)[2]
    ll = 0
    x̂ = y[:,1]
    Px = Pη
    for t in 2:T 
        dt = times[t]-times[t-1]
        x̂, Px, llt = ukf_update(y[:,t],x̂,Px,times[t],dt,f,pf,H,Pν,Pη,L,α,β,κ)
        ll += llt
    end
    return ll   
end


function init_kalman_loss(UDE::UDE,H,Pη,L,α,β,κ)
    y = UDE.data
    times = UDE.times
    f = (u,t,dt,p) -> UDE.process_model.predict(u,t,dt,p)[1]
    function loss(parameters)
        Pν = parameters.Pν * parameters.Pν'
       -1*ukf_likelihood(y,times,f,parameters.UDE.process_model,H,Pν,Pη,L,α,β,κ)
    end
    return loss
end

function mean_and_cov(x)
    μ = zeros(size(x)[1])
    Σ = zeros(size(x)[1],size(x)[1])
    for i in 1:size(x)[2]
        μ = μ .+ x[:,i]./size(x)[2]
    end 
    for i in 1:size(x)[2]
        Σ = Σ .+ ((x[:,i] .- μ) .* (x[:,i] .- μ)') ./(size(x)[2]-1)
    end 
    return μ, Σ
end


function init_single_kalman_loss(UDE::MultiUDE,H,Pη,L,α,β,κ)
    
    Imat = Matrix(I,L,L)
    function loss(parameters,data,series,starts,lengths)
        
        times = UDE.times[starts[series]:(starts[series]+lengths[series]-1)]
        y = data[:,starts[series]:(starts[series]+lengths[series]-1)]
        f = (u,t,dt,p) -> UDE.process_model.predict(u,series,t,dt,p)[1]

        Pν = Imat .* parameters.Pν.^2

        nll = -1*ukf_likelihood(y,times,f,parameters.UDE.process_model,H,Pν,Pη,L,α,β,κ)

        # regularization
        L_reg = UDE.process_regularization.loss(parameters.UDE.process_model,parameters.UDE.process_regularization)

        return nll + L_reg
    end

  
    return loss
end


function init_kalman_loss(UDE::MultiUDE,H,Pη,L,α,β,κ)
    
    single_loss = init_single_kalman_loss(UDE,H,Pη,L,α,β,κ)
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(UDE.data_frame, UDE.time_column_name, UDE.series_column_name)

    function loss(parameters)
        L = 0
        for i in eachindex(starts)
            L+= single_loss(parameters,data,i,starts,lengths)
        end
        return L
    end   

    return loss
end 



function ukf_smooth(UDE::MultiUDE,parameters,H,Pη,L,α,β,κ)

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(UDE.data_frame, UDE.time_column_name, UDE.series_column_name)

    x_ = zeros(size(UDE.data))
    Px_ = zeros(size(UDE.data)[1],size(UDE.data)[1],size(UDE.data)[2])
    Imat = Matrix(I,L,L)
    Pν = Imat .* parameters.Pν
    for series in eachindex(starts)
        times = UDE.times[starts[series]:(starts[series]+lengths[series]-1)]
        y = data[:,starts[series]:(starts[series]+lengths[series]-1)]
        f = (u,t,dt,p) -> UDE.process_model.predict(u,series,t,dt,p)[1]
        x,Px = ukf_smoothing(y,times,f,parameters.UDE.process_model,H,Pν,Pη,L,α,β,κ)
        x_[:,starts[series]:(starts[series]+lengths[series]-1)] .= x
        Px_[:,:,starts[series]:(starts[series]+lengths[series]-1)] .= Px
    end
   
    return x_, Px_
end 






function kalman_filter!(UDE::MultiUDE, σ2; H = -999, Pν = -999, Pη = -999, α = 10^-3, β = 2,κ = 0, verbose = true, maxiter = 500, step_size = 0.05)
    
    
    L = size(UDE.data)[1]
    if H == -999
        H = Matrix(I,L,L)
    end

    if Pν == -999
        Pν = σ2*ones(L)
    end 

    if Pη == -999
        Pη = σ2*Matrix(I,L,L)
    end


    loss = init_kalman_loss(UDE,H,Pη,L,α,β,κ)
    target = (x,u) -> loss(x)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    Pν_ = Pν
    params = ComponentArray((UDE = UDE.parameters, Pν = Pν_))
    optprob = Optimization.OptimizationProblem(optf, params)
    

    # print value of loss function at each time step 
    if verbose
        callback = function (p, l; doplot = false)
          print(round(l,digits = 3), " ")
          return false
        end
    else
        callback = function (p, l; doplot = false)
          return false
        end 
    end
  

    # run optimizer
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), callback = callback, maxiters = maxiter )
    

    # assign parameters to model 
    UDE.parameters = sol.u.UDE
    x, Px = ukf_smooth(UDE,sol.u,H,Pη,L,α,β,κ)
    UDE.parameters.uhat .= x
    Imat = Matrix(I,L,L)
    Pν = Imat .* sol.u.Pν

    return Pν, Px

end