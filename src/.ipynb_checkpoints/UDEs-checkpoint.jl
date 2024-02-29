# """
# This module defines the UDE model class

# This class (mutable struct) stores a process model that defines the dynamics of the system and observaiton model that defines the relationship between the state variables and the data and a regularization term. These three subcomponents are all instance of Abstract Types meaning they can be instances of of objects from a set of related classes that all have the same core functionality. 
# """
# module UDEs

include("ObservationModels.jl")
include("ProcessModels.jl")
include("LossFunctions.jl")
include("Regularization.jl")


"""
UDE

A data structure that stores the informaiton required to define a universal differntial equation model

times - the points in time when observaions are made
data - the obervaitons with each colum corresponding to a time point and each row a dimension of the observaitons vector
parameters - the wights and biases of neural netowrks and any other parameters estimated by the model
loss_function - a function that defines model performance that is minimized by the optimization routine
process_model - a function that predicts the evolution of the state variables between time points
process_loss - a funtion that quantifies the acuracy of the process model
observation_model - a function that descirbes the relationship between the observations and state variables
observations_loss - a function that describes the accuracy of the observation model
process_regularization - a function that penealizes the process model for complexty of adds prior information
observation_regularization - a function that penealizes the observation model for complexty of adds prior information
"""
mutable struct SSUDE
    times
    data
    data_frame
    parameters
    loss_function
    process_model
    process_loss 
    observation_model
    observation_loss 
    process_regularization
    observation_regularization
    constructor
end

"""
NeuralNet(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)

A constructor function that builds a UDE model for a given data set. This fuction builds a nonparamtric discrete time model 
for a time series. It assumes a 1 to 1 correspondence between the observations made int he data set and the underlying state
variables of the system and uses a neural network to describe the systems dynamics (state transitions).

The neural network has one hidden layer, but the number of hidden units and the seed used to initialize the paramters can be modified. 

data - A Julia DataFrame with one colum for time labels data.t and the rest allocated to state varaibles. 
hidden_units - the number of hidden units in the neural netowrk, defaults to 10 if no value is given
NN_seed -  the seed for the random number generator used to draw the neural network paramters, defualts to 1
proc_weight - the weight given to the accuracy of the process model in the loss function, defualts value of 1.0
obs_weight - the weight given to the accuracy of the observaiton model in the loss function, defualts value of 1.0
reg_weight - the weight given to regularizing the neural network in the loss function, defualts value of 10^-6
"""
function NeuralNet(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    
    # convert data
    data_frame = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data))
    
    process_model = ProcessModels.NeuralNetwork(size(data)[1];hidden = hidden_units, seed = NN_seed)
    process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    observation_model = ObservationModels.Identity()
    observation_loss = LossFunctions.MSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    # loss function 
    function loss_function(parameters)

        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end
    
        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            u1hat, epsilon = process_model.predict(u0,1.0,parameters.process_model) 
            L_proc += process_loss.loss(u1,u1hat,parameters.process_loss)
        end
    
        # regularization
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)
        
        return L_obs + L_proc + L_reg
    end
    
    constructor = data -> NeuralNet(data;hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    
    return SSUDE(times,data,data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                                observation_loss,process_regularization,observation_regularization,constructor)
    
end 



"""
DiscreteModelErrors(data, known_dynamics, init_known_dynamics_parameters; hidden_units=10, NN_seed = 1, errors_weight=0.1, MSE_weight=1.0, obs_weight=1.0, reg_weight = 0.001)

A constructor function that builds a discrete time  UDE model for a given data set and set of known dynamics. The known dynamcis are sepcified as a funtion with the same number of inputs and outputs as varaibles in the data set (excluding time). The model will jointly estimate the paramters of knwon dynamcis that are supplied to the argument `init_known_dynamics_parameters` and the paramters of a neural network that describe the knwon model errors. 

data - A julia DataFrame with one colum for time labels data.t and the rest allocated to state varaibles. 
known_dynamcis -  a julia function with the same number of inputs and outputs as varaibles in the data set
init_known_dynamics_parameters - a NamedTuple with an etry for each paramter estiamted by the model
hidden_units - the number of hidden units in the neural netowrk, defaults to 10 if no value is given
NN_seed -  the seed for the random number generator used to draw the neural network paramters, defualts to 1
errors_weight - the penelty for the neurla network predicting large residual terms 
MSE_weight - the weight given to the accuracy of the process model in the loss function, defualts value of 1.0
obs_weight - the weight given to the accuracy of the observaiton model in the loss function, defualts value of 1.0
reg_weight - the weight given to regularizing the neural network in the loss function, defualts value of 0.0
"""
function DiscreteModelErrors(data,known_dynamics,init_known_dynamics_parameters;
                                hidden_units=10,NN_seed = 1,errors_weight=0.1,MSE_weight=1.0,obs_weight=1.0,reg_weight = 0.0)
    
    # convert data
    data_frame = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data))
    
    process_model = ProcessModels.DiscreteModelErrors(size(data)[1],known_dynamics,init_known_dynamics_parameters;
                                                    hidden = hidden_units, seed = NN_seed)
    #process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    process_loss = LossFunctions.MSE_and_errors_sd(size(data)[1],N = size(data)[2]-1, errors_weight=errors_weight,MSE_weight=MSE_weight)
    observation_model = ObservationModels.Identity()
    observation_loss = LossFunctions.MSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L2_all(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    # loss function 
    function loss_function(parameters)

        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end
    
        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            u1hat, epsilon = process_model.predict(u0,1.0,parameters.process_model) 
            L_proc += process_loss.loss(u1,u1hat,epsilon,parameters.process_loss)
        end
    
        # regularization
        L_reg = process_regularization.loss(parameters.process_model.NN,parameters.process_regularization)
        
        return L_obs + L_proc + L_reg
    end
  
    constructor = data -> DiscreteModelErrors(data,known_dynamics,init_known_dynamics_parameters;
                                    hidden_units=hidden_units,NN_seed=NN_seed,errors_weight=errors_weight,
                                    MSE_weight=MSE_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    
    return SSUDE(times,data,data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 


"""
DiscreteModelErrors(data, known_dynamics; hidden_units=10, NN_seed = 1, errors_weight=0.1, MSE_weight=1.0, obs_weight=1.0, reg_weight = 0.001)

A constructor function that builds a discrete time UDE model for a given data set and set of known dynamics. The known dynamcis are sepcified as a funtion with the same number of inputs and outputs as varaibles in the data set (excluding time). The model estiamtes the paramters of a neural network that describe the known model errors. 

data - A julia DataFrame with one colum for time labels data.t and the rest allocated to state varaibles. 
known_dynamcis -  a julia function with the same number of inputs and outputs as varaibles in the data set
hidden_units - the number of hidden units in the neural netowrk, defaults to 10 if no value is given
NN_seed -  the seed for the random number generator used to draw the neural network paramters, defualts to 1
errors_weight - the penelty for the neurla network predicting large residual terms 
MSE_weight - the weight given to the accuracy of the process model in the loss function, defualts value of 1.0
obs_weight - the weight given to the accuracy of the observaiton model in the loss function, defualts value of 1.0
reg_weight - the weight given to regularizing the neural network in the loss function, defualts value of 0.0
"""
function DiscreteModelErrors(data,known_dynamics;hidden_units=10,NN_seed = 1,errors_weight=0.1,MSE_weight=1.0,obs_weight=1.0,reg_weight = 0.00)
    
    init_known_dynamics_parameters = NamedTuple()
    known_dynamics_function = (x,p) -> known_dynamics(x)
    
    return DiscreteModelErrors(data,known_dynamics_function,init_known_dynamics_parameters; hidden_units=hidden_units,NN_seed = NN_seed,errors_weight=errors_weight,MSE_weight=MSE_weight,obs_weight=obs_weight,reg_weight = reg_weight)
end 


"""
SSNODE(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)

A constructor function that builds a UDE model for a given data set. This fuction builds a nonparamtric continuous time model for a data set. It assumes a 1 to 1 correspondence between the observations made in the data set and the underlying state variables of the system and uses a neural ordinary differntial equaiton to describe the systems dynamics.

The neural network has one hidden layer, but the number of hidden units and the seed used to initialize the paramters can be modified. 

data - A Julia DataFrame with one colum for time labels data.t and the rest allocated to state varaibles. 
hidden_units - the number of hidden units in the neural netowrk, defaults to 10 if no value is given
NN_seed -  the seed for the random number generator used to draw the neural network paramters, defualts to 1
proc_weight - the weight given to the accuracy of the process model in the loss function, defualts value of 1.0
obs_weight - the weight given to the accuracy of the observaiton model in the loss function, defualts value of 1.0
reg_weight - the weight given to regularizing the neural network in the loss function, defualts value of 10^-7
"""
function SSNODE(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-7, l = 0.25,extrap_rho = 0.1 )
    
    # convert data
    data_frame = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data))
    
    process_model = ProcessModels.NODE_process(size(data)[1];hidden = hidden_units, seed = NN_seed, l = l, extrap_rho=extrap_rho)
    process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    observation_model = ObservationModels.Identity()
    observation_loss = LossFunctions.MSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    # loss function 
    function loss_function(parameters)

        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end
    
        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            u1hat, epsilon = process_model.predict(u0,dt,parameters.process_model) 
            L_proc += process_loss.loss(u1,u1hat,parameters.process_loss)
        end
    
        # regularization
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)
        
        return L_obs + L_proc + L_reg
    end
    
    constructor = (data) -> SSNODE(data;
                            hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return SSUDE(times,data,data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 



function CustomUDE(data, derivs;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-7)
    initial_parameters = NamedTuple()
    return CustomUDE(data,derivs,initial_parameters;hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
end
"""
ContinuousModelErrors(data, known_dynamics, init_known_dynamics_parameters; hidden_units=10, NN_seed = 1, errors_weight=0.1, MSE_weight=1.0, obs_weight=1.0, reg_weight = 0.001)

A constructor function that builds a continuous time UDE model for a given data set and set of known dynamics. The known dynamcis are sepcified as a funtion with the same number of inputs and outputs as varaibles in the data set (excluding time). The model will jointly estimate the paramters of known dynamcis that are supplied to the argument `init_known_dynamics_parameters` and the paramters of a neural network that describe the knwon model errors. 

data - A julia DataFrame with one colum for time labels data.t and the rest allocated to state varaibles. 
known_dynamcis -  a julia function with the same number of inputs and outputs as varaibles in the data set
init_known_dynamics_parameters - a NamedTuple with an etry for each paramter estiamted by the model
hidden_units - the number of hidden units in the neural netowrk, defaults to 10 if no value is given
NN_seed -  the seed for the random number generator used to draw the neural network paramters, defualts to 1
errors_weight - the penelty for the neurla network predicting large residual terms 
MSE_weight - the weight given to the accuracy of the process model in the loss function, defualts value of 1.0
obs_weight - the weight given to the accuracy of the observaiton model in the loss function, defualts value of 1.0
reg_weight - the weight given to regularizing the neural network in the loss function, defualts value of 0.0
"""
function ContinuousModelErrors(data,known_dynamics,init_known_dynamics_parameters;
                                    hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10.0^-7)
    
    # convert data
    data_frame = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data)).+1.0
    
    process_model = ProcessModels.ContinuousModelErrors(size(data)[1],known_dynamics,init_known_dynamics_parameters;
                                                            hidden = hidden_units, seed = NN_seed)
    process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    observation_model = ObservationModels.Identity()
    observation_loss = LossFunctions.MSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L1(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    # loss function 
    function loss_function(parameters)

        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end
    
        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            u1hat, epsilon = process_model.predict(u0,dt,parameters.process_model) 
            L_proc += process_loss.loss(u1,u1hat,parameters.process_loss)
        end
    
        # regularization
        L_reg = process_regularization.loss(parameters.process_model.NN,parameters.process_regularization)
        
        return L_obs + L_proc + L_reg
    end
    
    constructor = data -> ContinuousModelErrors(data,known_dynamics,init_known_dynamics_parameters;
                                    hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return SSUDE(times,data,data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 




function CustomDerivs(data,derivs,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,l=0.25,extrap_rho=0.1)
    
    # convert data
    dataframe = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,2:size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data))
    dims = size(data)[1]
    
    process_model = ProcessModels.ProcessModel_(derivs,ComponentArray(initial_parameters),dims,l=l,extrap_rho=0.1)
    process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    observation_model = ObservationModels.Identity()
    observation_loss = LossFunctions.MSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = initial_parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    # loss function 
    function loss_function(parameters)

        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end
    
        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            u1hat, epsilon = process_model.predict(u0,dt,parameters.process_model) 
            L_proc += process_loss.loss(u1,u1hat,parameters.process_loss)
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model.NN,parameters.process_regularization)
        
        return L_obs + L_proc + L_reg
    end
    
    
    constructor = (data) -> CustomDerivs(data,derivs,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return SSUDE(times,data,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end

function CustomDerivs(data, derivs;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-7)
    initial_parameters = NamedTuple()
    return CustomDerivs(data,derivs,initial_parameters;hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
end

"""
ContinuousModelErrors(data, known_dynamics; hidden_units=10, NN_seed = 1, errors_weight=0.1, MSE_weight=1.0, obs_weight=1.0, reg_weight = 0.001)

A constructor function that builds a continuous time UDE model for a given data set and set of known dynamics. The known dynamcis are sepcified as a funtion with the same number of inputs and outputs as varaibles in the data set (excluding time). The model estiamtes the paramters of a neural network that describe the known model errors. 

data - A julia DataFrame with one colum for time labels data.t and the rest allocated to state varaibles. 
known_dynamcis -  a julia function with the same number of inputs and outputs as varaibles in the data set
hidden_units - the number of hidden units in the neural netowrk, defaults to 10 if no value is given
NN_seed -  the seed for the random number generator used to draw the neural network paramters, defualts to 1
errors_weight - the penelty for the neurla network predicting large residual terms 
MSE_weight - the weight given to the accuracy of the process model in the loss function, defualts value of 1.0
obs_weight - the weight given to the accuracy of the observaiton model in the loss function, defualts value of 1.0
reg_weight - the weight given to regularizing the neural network in the loss function, defualts value of 0.0
"""
function ContinuousModelErrors(data,known_dynamics;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 0.0)
    
    known_dynamics1 = (u,p) -> known_dynamics(u)
    init_known_dynamics_parameters = NamedTuple()
    
    return ContinuousModelErrors(data,known_dynamics1,init_known_dynamics_parameters;
                                    hidden_units=hidden_units,NN_seed = NN_seed,proc_weight=proc_weight,
                                    obs_weight=obs_weight,reg_weight = reg_weight)
end 


"""
gradient_decent!(UDE::UDE; step_size = 0.05, maxiter = 500, verbos = false)

A funciton that optimizes the parameters of a UDE model using the ADAM gradeint decent algorithm. 
UDE - a UDE model 
step_size - the step size of the gradent decent algorithm, defaults to 0.05
maxiter - the maximum number of iterations to run the algorithm, defaults to 500
verbos- when true the function will print the value of the loss function at each iteration, defualts to false. 
"""
function gradient_decent!(UDE; step_size = 0.05, maxiter = 500, verbos = false)
    
    # set optimization problem 
    target = (x,p) -> UDE.loss_function(x)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)
    
    # print value of loss function at each time step 
    if verbos
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
    sol = Optimization.solve(optprob, OptimizationOptimisers.ADAM(step_size), callback = callback, maxiters = maxiter )
    
    # assign parameters to model 
    UDE.parameters = sol.u
    
    return nothing
end



function BFGS!(UDE; verbos = true, initial_step_norm = 0.01)
    
    if verbos
        callback = function (p, l; doplot = false)
          print(round(l,digits = 3), " ")
          return false
        end
    else
        callback = function (p, l; doplot = false)
          return false
        end 
    end
    
    
    target = (x,p) -> UDE.loss_function(x)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)

    sol = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = initial_step_norm);
        callback, allow_f_increases = false)

    # assign parameters to model 
    UDE.parameters = sol.u
    
end 


"""
plot_state_estiamtes(UDE::UDE)

Plots the value of the state variables estiamted by the UDE mdel 

UDE - a UDE model object
"""
function plot_state_estiamtes(UDE::SSUDE)
    
    plots = []
    for dim in 1:size(UDE.data)[1]
    
        plt=Plots.scatter(UDE.times,UDE.data[dim,:], label = "observations")
        
        Plots.plot!(UDE.times,UDE.parameters.uhat[dim,:], color = "grey", label= "estimated states",
                    xlabel = "time", ylabel = string("x", dim))
       
        push!(plots, plt)
    end 
            
    return plot(plots...)       
end 

function get_final_state(UDE::SSUDE)
    return UDE.parameters.uhat[:,end]
end 

"""
print_parameter_estimates(UDE::UDE)

prints the value of the known dynamcis paramters. 

UDE - a UDE model object
"""
function print_parameter_estimates(UDE::SSUDE)
    println("Estimated parameter values: ")
    i = 0
    for name in keys(UDE.parameters.process_model.known_dynamics)
        i += 1
        println(name, ": ", round(UDE.parameters.process_model.known_dynamics[i], digits = 3))
            
    end 
end


function predictions(UDE::SSUDE)
 
    inits = UDE.parameters.uhat[:,1:(end-1)]
    obs = UDE.parameters.uhat[:,2:end]
    preds = UDE.parameters.uhat[:,2:end]
    
    for t in 1:(size(inits)[2])
        u0 = inits[:,t]
        u1 = obs[:,t]
        dt = UDE.times[t+1] - UDE.times[t]
        preds[:,t] = UDE.process_model.predict(u0,dt,UDE.parameters.process_model)[1]
    end

    return inits, obs, preds
end 


"""
plot_predictions(UDE::UDE)

Plots the correspondence between the observed state transitons and the predicitons of the process model. 

UDE - a UDE model object
"""
function plot_predictions(UDE::SSUDE)
 
    inits, obs, preds = predictions(UDE)
    
    plots = []
    for dim in 1:size(obs)[1]
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "45 degree")
        scatter!(difs,preds[dim,:].-inits[dim,:],color = "white", label = "", xlabel = "Observed change Delta hatu_t", 
                                ylabel = "Predicted change hatut - hatu_t")
        push!(plots, plt)
            
    end
        
    return plot(plots...)
end


function max_(x)
    x[argmax(x)]
end

function min_(x)
    x[argmin(x)]
end

function mean_(x)
    sum(x)/length(x)
end

function forecast(UDE::SSUDE, u0::AbstractVector{}, times::AbstractVector{})
    
    uhats = UDE.parameters.uhat
    
    umax = mapslices(max_, UDE.parameters.uhat, dims = 2);umax=reshape(umax,length(umax))
    umin = mapslices(min_, UDE.parameters.uhat, dims = 2);umin=reshape(umin,length(umin))
    umean = mapslices(mean_, UDE.parameters.uhat, dims = 2);umean=reshape(umean,length(umean))
    
    
    #estimated_map = (x,dt) -> UDE.process_model.forecast(x,dt,UDE.parameters.process_model,umax,umin,umean)
    estimated_map = (x,dt) -> UDE.process_model.forecast(x,dt,UDE.parameters.process_model,umax,umin,umean)
    
    
    x = u0
    df = zeros(length(times),length(x)+1)
    df[1,:] = vcat([times[1]],x)
    
    for t in 2:length(times)
        dt = times[t]-times[t-1]
        x = estimated_map(x,dt)
        df[t,:] = vcat([times[t]],x)
    end 
    
    return df
end 


"""
plot_forecast(UDE::UDE, T)

Plots the models forecast up to T time steps into the future from the last observaiton.  

UDE - a UDE model object
T - the nuber of time steps to forecast
"""
function plot_forecast(UDE::SSUDE, T)
    u0 = UDE.parameters.uhat[:,end]
    dts = UDE.times[2:end] .- UDE.times[1:(end-1)]
    dt = sum(dts)/length(dts)
    times = UDE.times[end]:dt:(UDE.times[end] + T*dt )
    df = forecast(UDE, u0, times)
    plots = []
    for dim in 2:size(df)[2]
        plt = plot(df[:,1],df[:,dim],color = "grey", linestyle=:dash, label = "forecast",
                    xlabel = "Time", ylabel = string("x", dim))
        plot!(UDE.times,UDE.data[dim-1,:],c=1, label = "data",
                    xlabel = "Time", ylabel = string("x", dim))
        push!(plots, plt)
    end 
    return plot(plots...), plots
end 



function forecast_simulation_test(simulator,model,seed;train_fraction=0.9,step_size = 0.05, maxiter = 500)
    
    # generate data and split into training and testing sets 
    data = simulator(seed)
    N_train = floor(Int, train_fraction*size(data)[1])
    train_data = data[1:N_train,:]
    test_data = data[(N_train):end,:]
    
    # build model 
    model = model.constructor(train_data)
    UDEs.gradient_decent!(model, step_size = step_size, maxiter = maxiter) 
    
    # forecast
    u0 = get_final_state(model)
    times = test_data.t
    predicted_data = forecast(model, u0, times)
    predicted_data= DataFrame(predicted_data,names(test_data))
    
    # MSE
    SE = copy(predicted_data)
    SE[:,2:end] .= (predicted_data[:,2:end] .- test_data[:,2:end]).^2
    return train_data, test_data, predicted_data , SE
end 

function forecast_simulation_SE(simulator,model,seed;train_fraction=0.9,step_size = 0.05, maxiter = 500)
    
    # generate data and split into training and testing sets 
    data = simulator(seed)
    N_train = floor(Int, train_fraction*size(data)[1])
    train_data = data[1:N_train,:]
    test_data = data[(N_train):end,:]
    
    # build model 
    model = model.constructor(train_data)
    UDEs.gradient_decent!(model, step_size = step_size, maxiter = maxiter) 
    
    # forecast
    u0 = get_final_state(model)
    times = test_data.t
    predicted_data = forecast(model, u0, times)

    return (Matrix(predicted_data[:,2:end]) .- Matrix(test_data[:,2:end])).^2
end 



function forecast_simulation_tests(N,simulator,model;train_fraction=0.9,step_size = 0.05, maxiter = 500)
    
    # get test data size and set accumulator
    sizeSE = forecast_simulation_SE(simulator,model,1;train_fraction=train_fraction,step_size = step_size, maxiter = 1)
    
    MSE_acc = [zeros(size(sizeSE)) for i in 1:Threads.nthreads()]
    
    # run simulation tests with multithreading
    Threads.@threads for seed in 1:N
            
        MSE_acc[Threads.threadid()] .+= forecast_simulation_SE(simulator,model,seed;train_fraction=train_fraction,step_size = step_size, maxiter = maxiter) ./ N
            
    end 
        
    MSE = MSE_acc[1]
    for i in 2:Threads.nthreads()
        MSE .+= MSE_acc[i]
    end 
                
    T = size(MSE)[1]
    MSE = DataFrame(MSE ,:auto)
    MSE.t = 1:T
                
    return MSE
  
end 



function leave_future_out(model; forecast_length = 10,  forecast_number = 10, spacing = 1, step_size = 0.05, maxiter = 500)
    
    # get final time
    data = model.data_frame
    T = length(data.t)
    start1 = T - forecast_length - spacing*(forecast_number-1)
    starts = [start1 + spacing *i for i in 0:(forecast_number-1)]
    training_data = [data[1:t0,:] for t0 in starts]
    testing_data = [data[t0:(t0+forecast_length),:] for t0 in starts]
    
    standard_errors = [[] for i in 1:Threads.nthreads()]
    predicted = [[] for i in 1:Threads.nthreads()]
    
    Threads.@threads for i in 1:forecast_number
        
        model_i = model.constructor(training_data[i])
                        
        gradient_decent!(model_i, step_size = step_size, maxiter = maxiter) 
                        
        try
            BFGS!(model_i)
        catch
            gradient_decent!(model_i, step_size = 0.25*step_size, maxiter = maxiter)                 
        end                    
                    
        # forecast
        u0 = get_final_state(model_i)
        times = testing_data[i].t
        predicted_data = forecast(model_i, u0, times)
        predicted_data= DataFrame(predicted_data,names(testing_data[i]))
            
        SE = copy(predicted_data)
        SE[:,2:end] .= (predicted_data[:,2:end] .- testing_data[i][:,2:end]).^2
        
        push!(standard_errors[Threads.threadid()], SE)
        push!(predicted[Threads.threadid()], predicted_data)             
    end 
    
    standard_error = standard_errors[1]
    predicted_data = predicted[1]         
    for i in 2:Threads.nthreads()
                            
        standard_error = vcat(standard_error,standard_errors[i])
        predicted_data = vcat(predicted_data,predicted[i])
                            
    end
    
    return training_data, testing_data, standard_error, predicted_data
    
end 


function leave_future_out_mse(standard_errors)
    N = length(standard_errors)
    acc = zeros(size(Matrix(standard_errors[1])[:,2:end]))
        
    for i in 1:N
                                    
        acc .+= standard_errors[i][:,2:end] ./ N
                                    
    end 
    
    MSE = DataFrame(hcat(collect(1:size(acc)[1]), acc), names(standard_errors[1]))
    
    return MSE
end 


function leave_future_out_cv(model; forecast_length = 10,  forecast_number = 10, spacing = 1, step_size = 0.05, maxiter = 500)
    training_data, testing_data, standard_errors, predicted_data = leave_future_out(model;forecast_length =forecast_length,forecast_number=forecast_number, spacing=spacing,step_size=step_size,maxiter=maxiter)
    MSE = leave_future_out_mse(standard_errors)
    return MSE, training_data, testing_data, standard_errors , predicted_data                                   
end

                                        
function plot_leave_future_out_cv(data,testing_data, standard_errors , predicted_data)
    plts1 = []
    plts2 = []
    for i in 2:(size(data)[2])
        p1=Plots.scatter(data.t,data[:,i])
        p2 = Plots.plot([0,data.t[end]],[0.0,0.0], color = "black", linestyle = :dash)
        for j in 1:length(testing_data)
            Plots.plot!(p1,predicted_data[j].t,predicted_data[j][:,i], linestyle = :dash, width= 2) 
            Plots.plot!(p2,standard_errors[j].t,standard_errors[j][:,i], width= 2)
        end 
        push!(plts1,p1)
        push!(plts2,p2)
    end
    p1 = plot(plts1...)
    p2 = plot(plts2...)
    return p1,p2
end 
# end # module