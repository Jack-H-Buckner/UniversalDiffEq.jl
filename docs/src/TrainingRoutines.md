# Training routines (This section is incomepte see the docuemtnation for the train! funiton in the API)


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

State-space loss functions assume the data $y_t \in R^d$ can be described by a set of state variables $u_t \in R^d$ and observation errors $\epsilon_t \in R^d$. The UDE model $f(u,X(t),t|\theta)$ with paramters $\theta$ and covariates $X$ is used to learn the dynamcis of the state varibles $u_t$.  Combining these assumptions yield two equations that descibe the observation set $y_t$

$$
y_t = u_t + \epsilon_t \\ 
u_{t+1} = u_t + \int_{t}^{t+1} f(u,X(v),v|\theta) dv + \nu_t
$$

Given these two equations we can calculate the likelihood of the data $y_t$ given the parameters of the UDE $\theta$ in two ways, the conditional and marginal likeihood. The conditional likelihood is used in bayesian settings and describes the likelihood of the data given the UDE paramters $\theta$ and point estimates of the state variables $\hat{u}_t$. The marginal likelihood desribes the likelihood of the observaitons $y_t$ given the model parameters $\theta$ while accounting for uncertianty in the estimates of the state variables $u_t$. The marginal likelihood is used in frequenties setting and produces unbiased maximum likelihood estiamtes of the paramters $\theta$. In paractice, both likelihood functions can be used to train the UDE models with good results. The conditional likeihood is more compuationally efficent, while the marginal likelihood is, in theory, more accurate. 

#### Conditional likelihood
We calcualte the conditional likelihood by jointly estimating the UDE model parameters $\theta$ and the value of the underlying state variables $\hat{u}_t$. We assume that the process $\nu_{i,t}$ and observation errors $\epsilon_{i,t}$ along each dimesnon of the data set $i\in \{1:d\}$, are independnet identically distributed normal random variables with variances $\tau^2$ and $\sigma^2$ respectively. 

Given this assumption the log likelihood function has two components, a term corresponding to the observation errors, and term corresponding to the process process errors. The observation error term is the sum of the mean squared observaiton errors scaled by the observaiton variance  $\sigma^2$

$$
L_{obs}(\hat{u}) = \sum_{t=1}^{T} \sum_{i=1}^{d} \left( \frac{y_{i,t}-\hat{u}_{i,t}}{\sigma}\right)^2
$$

The process error term has the same stucture but it calcautle the sum of squarred process errors

$$
L_{proc}(\hat{u},\theta) = \sum_{t=1}^{T} \sum_{i=1}^{d} \left( \frac{\hat{u}_{i,t+1} - F(\hat{u}_t,t,\theta)}{\tau}\right)^2, 
$$

where

$$
F(u,\theta) = u + \int_{t}^{t+1} f(u,X(v),v|\theta) dv + \nu_t.
$$

Combining these terms yields the log conditiona likeihood function 

$$
L(\hat{u},\theta) = L_{obs}(\hat{u}) + L_{proc}(\hat{u},\theta).
$$

#### Marginal
We calcualte the marginal likeihood using an unscented kalman filters. 

$$
\Pi_{t=1}^{T}\int_{-\infty}^{\infty} \left(\frac{y_t-u_t}{\sigma}\right)^2f(u_t;\{y\}_{1:(t-1)},\theta)
$$

### Shooting

### Multiple shooting





