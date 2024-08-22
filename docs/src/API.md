# API

## Model Functions  
List of model functions
```@docs
UniversalDiffEq.BayesianCustomDerivatives(data,derivs!,initial_parameters;kwargs ... )
UniversalDiffEq.BayesianCustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;kwargs ... )
UniversalDiffEq.BayesianCustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )
UniversalDiffEq.BayesianNODE(data;kwargs ... )
UniversalDiffEq.BayesianNODE(data,X;kwargs ... )
UniversalDiffEq.CustomDerivatives(data,derivs!,initial_parameters;kwargs ... )
UniversalDiffEq.CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )
UniversalDiffEq.CustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;kwargs ... )
UniversalDiffEq.CustomDifference(data,step,initial_parameters;kwrags...)
UniversalDiffEq.CustomDifference(data::DataFrame,X,step,initial_parameters;kwargs ... )
UniversalDiffEq.CustomDifference(data::DataFrame,step,initial_parameters,priors::Function;kwargs ... )
UniversalDiffEq.EasyNODE(data,X;kwargs ... )
UniversalDiffEq.EasyNODE(data;kwargs ... )
UniversalDiffEq.EasyUDE(data,derivs!,initial_parameters;kwargs ... )
UniversalDiffEq.EasyUDE(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )
UniversalDiffEq.MultiNODE(data;kwargs...)
UniversalDiffEq.MultiNODE(data,X;kwargs...)
UniversalDiffEq.NNDE(data;kwargs ...)
UniversalDiffEq.NODE(data;kwargs ... )
UniversalDiffEq.NODE(data,X;kwargs ... )
```

## Analysis Functions  
Below are all functions included for model analysis and description:

```@docs
UniversalDiffEq.equilibrium_and_stability(UDE,lower,upper;t=0,Ntrials=100,tol=10^-3)
UniversalDiffEq.equilibrium_and_stability(UDE,X,lower,upper;t=0,Ntrials=100,tol=10^-3)
UniversalDiffEq.equilibrium_and_stability(UDE::MultiUDE,site,X,lower,upper;t=0,Ntrials=100,tol=10^-3)
UniversalDiffEq.get_right_hand_side(UDE::UDE)
UniversalDiffEq.get_NN_parameters(UDE::UDE)
UniversalDiffEq.get_parameters(UDE::UDE)
UniversalDiffEq.kfold_cv(model::UDE)
UniversalDiffEq.leave_future_out_cv(model::UDE)
UniversalDiffEq.phase_plane(UDE::UDE; idx=[1,2], u1s=-5.0,0.25,5.0, u2s=-5:0.25:5,u3s = 0,T = 100)
UniversalDiffEq.phase_plane(UDE::UDE, u0s::AbstractArray; idx=[1,2],T = 100)
UniversalDiffEq.phase_plane_3d(UDE::UDE; idx=[1,2,3], u1s=-5.0,0.25,5.0, u2s=-5:0.25:5,u3s=-5:0.25:5,T = 100)
UniversalDiffEq.plot_forecast(UDE::UDE, T::Int)
UniversalDiffEq.plot_forecast(UDE::UDE, test_data::DataFrame)
UniversalDiffEq.print_parameter_estimates(UDE::UDE)
UniversalDiffEq.plot_predictions(UDE::UDE)
UniversalDiffEq.plot_predictions(UDE::UDE, test_data::DataFrame)
UniversalDiffEq.plot_state_estimates(UDE::UDE)
```