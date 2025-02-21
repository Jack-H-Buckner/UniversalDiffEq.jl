# Training routines


There are many differnt methods for training NODE and UDE models. These methods tradeoff between accuracy, stability, and computing time. Their performance may also be related to the characteristics of the training data. However, this is an active area of research where it is difficult to make definitive statements. 


Currently, UniversalDiffEq.jl implements fixed loss functions and two different optimization algorithms that can be accessed through the `train!` function. The logic behind each method and details of implementation can be found below.


## loss functions 


### Derivative matching 

Derivative matching is the most computationally efficient training procedure implemented by UniversalDiffEq.jl. It works for continuous time models such as those implemented by `CustomDerivatives` or `NODE` model building functions. This method was adapted from a tutorial in the DiffEqFlux.jl (documentation)[https://docs.sciml.ai/DiffEqFlux/stable/examples/collocation/] called smooth collocation for fast two-stage training.

Derivative matching trains the models using a two-step procedure. First, the algorithm fits a smoothing curve $s_i(t)$ to each dimension $i$ of the time series using a cubic spline implemented by the DataInterpolations.jl package. The model is trained by comparing the right-hand side $f$ of the UDE model to the derivatives of the smoothing curve, evaluated at the time of each observation in the data set. 

```math
L(\theta) = \sum_i \sum_{\tau\in T} \left(\frac{ds_i}/{dt} |_\tau - f_i(s(\tau)\rigt)^2 + \omega_R |\theta_{w}|_{L2} 
``` 
The final term $\omega_R |\theta_{w}|_{L2} $ applies $L2$ regualrization to the neural network weights $\theta_{w}$. The user can specify the weight $\omega_R$ using the keyword argument `Regularization_weight` in the `train!` function.

The algorithm can be tuned for a specific data set using the `loss_options` keyword argument. This argument should be a NamedTyple with values `d` and `remove_ends`. The parameter `d` sets the number of degrees of freedom used by the curve-fitting model. The `remove_ends` option is an integer. This allows data points from the beginning and end of the data set to be excluded from the loss function. The default value is zero (no observations are excluded), but the smoothing curves might fit poorly near the beginning and end of some data sets. Setting `remove_ends > 0` can help reduce the influence of these edge effects on the trained model.The smoothing curves are fit using the generalized cross-validation method in the (DataInterpolations.jl)[https://docs.sciml.ai/DataInterpolations/dev/methods/#Regularization-Smoothing] package. This method fits a continous differntiable curve to the data with a smoothing penelty term chosen to minimize the influecne of observaiton errors. 

The success of the training procedure can be evaluated using the `plot_state_estimates` and `plot_predictions` functions. The `plot_state_estimates` will will plot the data using a scatter plot and over lay the smoothing curves with a line plot allowing a visual inspection of the fits. The plot predictons function will show who well the trained UDE predictes the changes in the smoothing curves forecasting one step ahead.  

### State-space loss functions

State-space loss functions assume the data include observation errors and that the underlying dynamics cannot be perfectly predicted by the UDE model. These state-space training rotines account for these sources of unceritainty by embedding the UDE model into a state-space modeling framework.  

### State-space conditional likelihood

### Shooting

### Multiple shooting





