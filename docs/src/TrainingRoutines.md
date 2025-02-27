 # Training routines


There are many differnt methods for training NODE and UDE models. These methods trade off between accuracy, stability, and computing time. Their performance may also be related to the characteristics of the training data. However, this is an active area of research where it is difficult to make definitive statements.


Currently, UniversalDiffEq.jl implements five loss functions and two different optimization algorithms that can be accessed through the `train!` function. The logic behind each method and the implementation details can be found below.


## Gradient matching loss function
### Method description
Gradient matching is the most computationally efficient training procedure implemented by UniversalDiffEq.jl. It works for continuous time models such as those implemented by `CustomDerivatives` or `NODE` model-building functions. This method was adapted from a tutorial in the DiffEqFlux.jl (documentation)[https://docs.sciml.ai/DiffEqFlux/stable/examples/collocation/] called smooth collocation for fast two-stage training.


Gradient matching trains the models using a two-step procedure. First, the algorithm fits a smoothing curve $s_i(t)$ to each dimension $i$ of the time series using a cubic spline implemented by the DataInterpolations.jl package. The model is trained by comparing the right-hand side $f$ of the UDE model to the derivatives of the smoothing curve, evaluated at the time of each observation in the data set.


```math
   L(\theta) = \sum_i \sum_{\tau\in T} \left(\frac{ds_i}{dt} |_\tau - f_i(s(\tau)\right)^2 + \omega_R |\theta_{w}|_{L2}
```
The final term $\omega_R |\theta_{w}|_{L2} $ applies $L2$ regualrization to the neural network weights $\theta_{w}$. The user can specify the weight $\omega_R$ using the keyword argument `Regularization_weight` in the `train!` function.


The algorithm can be tuned for a specific data set using the `loss_options` keyword argument. This argument should be a NamedTyple with values `d` and `remove_ends`. The parameter `d` sets the number of degrees of freedom used by the curve-fitting model. The `remove_ends` option is an integer. This allows data points from the beginning and end of the data set to be excluded from the loss function. The default value is zero (no observations are excluded), but the smoothing curves might fit poorly near the beginning and end of some data sets. Setting `remove_ends > 0` can help reduce the influence of these edge effects on the trained model.The smoothing curves are fit using the generalized cross-validation method in the (DataInterpolations.jl)[https://docs.sciml.ai/DataInterpolations/dev/methods/#Regularization-Smoothing] package. This method fits a continuous, differentiable curve to the data with a smoothing penalty term chosen to minimize the influence of observaiton errors.


The success of the training procedure can be evaluated using the `plot_state_estimates` and `plot_predictions` functions. The `plot_state_estimates` will plot the data using a scatter plot and overlay the smoothing curves with a line plot, allowing a visual inspection of the fits. The plot predictions function will show how well the trained UDE predicts the changes in the smoothing curves ,forecasting one step ahead. 




### Implementation


The gradient matching loss function is implemented using the `train` function by setting the key work argument `loss_function="gradient matching`.  The behavior of the loss function can be modified by supplying a value for the `regularization_weight` weight by providing the degrees of freedom `d` and edge effects `remove_ends` using the `loss_options` argument. The block of Julia code below shows the default values for each argument.


```julia
train!(model;
       loss_function = "gradient matching",
       regularization_weight = 0.0,
       loss_options = (d = 12, remove_ends = 0)
   )
```


Note that if only one value is provided to `loss_options` it needs to be followed by a comma (e.g. `loss_options=(d = 12,)`)
## State-space loss functions


State-space loss functions assume the data $y_t \in R^d$ can be described by a set of state variables $u_t \in R^d$ and observation errors $\epsilon_t \in R^d$. The UDE model $f(u,X(t),t|\theta)$ with parameters $\theta$ and covariates $X$ is used to learn the dynamics of the state variables $u_t$.  Combining these assumptions yields two equations that describe the observation set $y_t$


```math
   y_t = u_t + \epsilon_t \\
   u_{t+1} = u_t + \int_{t}^{t+1} f(u,X(v),v|\theta) dv + \nu_t
```


Given these two equations, we can calculate the likelihood of the data $y_t$ given the parameters of the UDE $\theta$ in two ways, the conditional and marginal likelihood. The conditional likelihood is used in Bayesian settings and describes the likelihood of the data given the UDE parameters $\theta$ and point estimates of the state variables $\hat{u}_t$. The marginal likelihood describes the likelihood of the observations $y_t$ given the model parameters $\theta$ while accounting for uncertianty in the estimates of the state variables $u_t$. The marginal likelihood is used in a frequenties setting and produces unbiased maximum likelihood estimates of the parameters $\theta$. In practice, both likelihood functions can be used to train the UDE models with good results. The conditional likelihood is more computationally efficient, while the marginal likelihood is more accurate in theory.


### Conditional likelihood
To start, we need to specify a family of distributions for the observaiton and process errors. Our implementation assumes these terms follow multivariate normal distributions with mean zero and covariance $\Sigma_{proc}$ and $\Sigma_{obs}$. The covariance matrices must be supplied by the user for the conditional likelihood approach.


Given the distributional assumptio, the log-likelihood function has two components, a term corresponding to the observation errors and a term corresponding to the process errors. The observaiton error term comparte the differnce between the obervations $y_t$ and estimated states $\hat{u}_t$ weighted by the observaiton error matrix $\Sigma_{obs}$


```math
   L_{obs}(\hat{u}) = \sum_{t=1}^{T} \sum_{i=1}^{d} \left(y_{i,t}-\hat{u}_{i,t}\right)^{T} \Sigma_{obs}^{-1} \left(y_{i,t}-\hat{u}_{i,t}\right)
```


The process error term has the same structure but compares the UDE model predictions $F(\hat{u}_t,t,\theta)$ to the estimated states $\hat{u}_{i,t+1}$


```math
   L_{proc}(\hat{u},\theta) = \sum_{t=1}^{T} \sum_{i=1}^{d} \left(\hat{u}_{i,t+1} - F(\hat{u}_t,t,\theta)\right)^{T} \Sigma_{proc}^{-1}\left(\hat{u}_{i,t+1} - F(\hat{u}_t,t,\theta)\right)
```






### Marginal likelihood
The marginal likelihood calculates the likelihood of the data $y_t$ given the UDE model parameters $\theta$ by estimating the state variables and the associated uncertainty. This requires distributional assumptions about the observation and process errors. As with the conditional likelihood, our implementation assumes these distributions are multivariate normal. However, the marginal likelihood only requires the user to provide the observaiton error matrix $\Sigma_{obs}$ because the process errors $\Sigma_{obs}$ are estimated as part of the model training routine.


Given these distributional assumptions, the likelihood is calculated by integrating over the distribution of the states $u_t$ to get a quantity that only depends on the data $y_t$ and the parameters $\theta$. We approximate the distribution of the state variables $u_t$ using the [unsented kalman filter algorithm](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf). A detailed description of the algorithm can be found in the proceeding link; the following description focuses on explaining how we use the algorithm to calculate the likelihood.


The algorithm approximates the distribution of the state variables $u_t$ at each time step with a multivariate normal distribution with mean $\hat{u}_t$ and covariance $\Sigma_{u,t}$. The values of $\hat{u}_t$ and $\Sigma_{u,t}$ are calcualted by an iterative process that starts with an inital estimate of the mean $\hat{u}_0$ and covariance $\Sigma_{u,0}$ at the time of the first observation. The estimates at the subsequent time steps are calculated by iteratively applying a twp step process. First, the initial estimate $\hat{u}_{t}$ and  $\Sigma_{u,t}$ is updated by conditioning on the observaiton $y_t$. Because we assume the initial distirubtion of $u_{t}$ and the observaiton erros are multivariate normal this update step has a close form [solution](https://en.wikipedia.org/wiki/Kalman_filter), appying these fomulas yields updated estimates $\hat{u}_{t}`$ and  $\Sigma_{u,t}`$. The next uses the UDE model for forecast the value state variables in the next time step $\hat{u}_{t+1}$ and uses the unscented transform to propagate uncertainty through the UDE model. This error propogation step yields an estimate of the uncertianty accounting for the deterministic component of the dynamics  $Cov[\hat{u}_t]$. The final unceritnaty estimate is the propogated uncertainty plus the process errors $\Sigma_{u,t+1} = Cov[\hat{u}_t] + \Sigma_{proc}$.


Given our approximation of the states $u_t$ as multivariate normal, the distribution of $y_t$ is the sum of two multivariate normal random variables (the state estimates $u_t$ and the observation errors $\epsilon_{t}$). Therefore, the marginal likelihood is a multivariate normal with mean $u_t$ and covariance $\Sigma_{u,t} + \Sigma_{obs}$. Taking the log and summing over each data point in the time series yields the marginal likelihood function of the model.


```math
L(\theta,\Sigma_{proc}) = \sum_{t=1}^{T} (y_t - \hat{u}_t)^{T} (\Sigma_{u,t} + \Sigma_{obs})^{-1} (y_t - \hat{u}_t) - 1/2 log(|(\Sigma_{u,t} + \Sigma_{obs})|) - d/2log(2\pi).
```
where $|\Sigma|$ is the determiniant of the matrix $\Sigma$ and $\Sigma^{-1}$ is the matrix inverse.


### Implementation


The state-space loss functions can be accessed through the `train!` function by setting the `loss_function` keyword argument to `"conditional likelihood"` or `"marginal likelihood"`. The user can also provide a value for the regularization weight and set the process and observaiton error matrices using the `loss_options` argument. The process errors are set using the `process_error` key and observaiton errors using the `observation_error` key. The user can supply a `Float`, a vector of length d or a positive definite d$\times$d matrix. If a `Float` is provided, the error matrix will use that value along the diagonal and set all covariance terms to zero. If a vector is provided, it will be used as the diaganol of the matrix with all other terms equal to zero, and if a matrix is provided, it will be used as the full error covariance matrix.


```julia
# Conditional likelihood
train!(model;
       loss_function = "conditional likelihood",
       regularization_weight = 0.0,
       loss_options = (process_error = 0.025,observation_error = 0.025)
   )


# Marginal likelihood
train!(model;
       loss_function = "marginal likelihood",
       regularization_weight = 0.0,
       loss_options = (process_error = 0.025,observation_error = 0.025)
   )
```


Note that if only one value is provided to `loss_options` it needs to be followed by a comma (e.g. `loss_options=(process_error = 0.025,)`)


## Shooting loss function


The shooting loss function jointly estimates the initial conditions $u_0$ and the UDe model parameters $\theta$ by numerically solving the UDE model starting $u_0$ over the full length of the training data set to get a simulated trajectory $\hat{u}(t)$. The loss is calculated by comparing the simulated trajectory to the data set with the mean squared error


```math
   L(u_0,\theta)= \frac{1}{d*T} \sum_{t=0}^{T} \sum_{i=1}{d}(y_{i,t} - \hat{u}_i(t))^2.
```


This method can work for data sets with relatively smooth changes over time but is highly susceptible to getting stuck at local minimum solutions on data sets with oscillations. This method is totally unsuitable for systems with chaotic dynamics because of the sensitivity of the simulation trajectories to the initial conditions and model parameters.


### Implementation


The shooting loss is accessed through the `train!` function by setting `loss_function = "shooting"` The regualrization weight can be set, but there are no additional arguments for this method.
```julia
train!(model;
       loss_function = "shooting",
       regularization_weight = 0.0
   )
```


## Multiple shooting loss function


The multiple shooting loss function tries to solve the issues with the shooting loss function by breaking that data set up into $k$ segments. The training routine estimates a set of initial conditions $u_{\tau}$ corresponding to the initial time point in each segment of the data set. The solution of the UDE model is solved numerically over each interval to create a set of $k$ simulated trajectories associated with each starting point $\hat{u}_k(t)$.  The loss is calculated by comapring


```math
   L(\{u\}_{\tau},\theta)= \frac{1}{d*T} \sum_{t=0}^{T} \sum_{i=1}{d}(y_{i,t} - \hat{u}_{\tau,i}(t))^2.
```


This approach can help remove the local minimum solutions that plague the shooting loss function, provided a large enough number of segments $k$ are chosen.




### Implementation
The shooting loss is accessed through the `train!` function by setting `loss_function = "multiple shooting"` The regualrizaiton weight can be set using the  `regularization_weight` argument, and the number of data point spaned by each prediction interval is set using `loss_options = (pred_length = n,)`. The default value of pred_length is five


```julia
train!(model;
       loss_function = "multiple shooting",
       regularization_weight = 0.0,
       loss_options = (pred_length = 5,)
   )
```


## ADAM optimizer


The (Adam)[chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1412.6980] optimizer is a gradient descent algorithm that has two parameters, `step_size` and  `maxiter`. The defaults for these parameters are set to values that perform well with each loss function. Increasing the maximum number of iterations `maxiter` often improves model fits.


```julia
train!(model;
       optimizer = "ADAM",
       verbose = true,
       optim_options = (step_size = 0.05, maxiter = 500)
   )
```


## BFGS optimizer


The L-BFGS is an alternative method that uses approximate second-order information to minimize the loss function. This method can improve model fits when compared to Adam but can also lead to overfitting if the neural network is not sufficiently regularized


```julia
train!(model;
       optimizer = "BFGS",
       verbose = true,
       optim_options = (initial_stepnorm = 0.01,)
   )
```



