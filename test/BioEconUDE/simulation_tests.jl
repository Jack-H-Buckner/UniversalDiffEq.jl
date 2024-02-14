
using Plots

function plot_sim_test(data,MSE, training_data, testing_data, standard_error, predicted_data)
    
    vars = size(data)[2]-1
    data_plots = [plot() for i in 1:vars]
    MSE_plots = [plot() for i in 1:vars]
    for i in 1:vars
        data_plots[i] = Plots.scatter(data.t,data[:,1+i], label = "data", xlabel = "time", ylabel = string("Var ", i)) 
        MSE_plots[i] = Plots.plot([0,data.t[end]],[0.0,0.0], color = "black", linestyle = :dash, label = "", xlabel = "time", ylabel = "MSE")
        if i == 1
            data_plots[i] = Plots.plot!(data_plots[i],[0],[0], linestyle = :dash, color = "grey", 
                                        label = "forcast")
        end
        
        for j in 1:length(testing_data)
            Plots.plot!(data_plots[i],predicted_data[j].t,predicted_data[j][:,1+i], linestyle = :dash, width= 2,label = "") 
            Plots.plot!(MSE_plots[i],standard_error[j].t,standard_error[j][:,1+i], width= 2, label = "")
        end 
        
    end 
 
    return plot(plot(data_plots...,layout = (vars,1)),plot(MSE_plots...,layout = (vars,1))) 
end


function simulation_test(data_generator, model,forecast_length,forecast_number,spacing,maxiter,stepsize)
    data = data_generator()
    model1 = model.constructor(data)
    
    MSE, training_data, testing_data, squared_errors, predicted_data = UniversalDiffEq.leave_future_out_cv(model1;forecast_length = forecast_length,  
                                forecast_number = forecast_number, spacing = spacing, step_size = stepsize, maxiter = maxiter)
    data_quantity = [length(training_data[i].t) for i in 1:length(training_data)]
    return squared_errors, data_quantity
end 



function simulation_tests(data_generator,model,forecast_length,forecast_number,spacing;Nsims = 10,maxiter=500,stepsize=0.05)
    
    SEs = zeros(forecast_number) 
    data_quantity = zeros(forecast_number)
    
    for i in 1:Nsims
        print(i, " ")
        SE, data_quantity = simulation_test(data_generator,model,forecast_length,forecast_number,spacing,maxiter,stepsize)
        for j in 1:forecast_number
            SEs[j] += sum(Matrix(SE[j][:,2:end]))/prod(size(SE[j][:,2:end])) 
        end 
    end 
    
    
    return sqrt.(SEs)./forecast_number, data_quantity
end 