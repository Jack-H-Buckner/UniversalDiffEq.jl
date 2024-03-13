# Examples

NODEs are nonparametric time series models that represent changes in the time series with a differntial equaiton where the righ hand side is a neural network
```math
\frac{du}{dt} = NN(u;w,b).
```
The dynamics of the resulting differntial equation are determined by the weights ``w`` and biases ``b`` of the neural network, which are found in the training process. 

The `UniversalDiffeq.jl` package is designed with ecologists in mind. Ecologogical data tends to be reletivly sparce and noisy compared to data on physical systems. To accomidate these limitations the training routine used by `UniversalDiffeq.jl` explicitly accounts for observaiton errors. We assume the data ``x_t`` is the sum of the true value of the systems state ``u_t`` plus a noise term ``\epsilon_t``. The training routine jointly estimates the value fo the state variables ``u_t`` and the weights ``w`` and biases ``b`` of the neural network by splitting the loss funciton into to components, and observation loss and a prediction loss. 

The observaiton loss compares the estimated state variables ``\hat{u}_t`` to the data``x_t`` using the mean squared error
```math
L_{observation} = \frac{1}{T}\sum_{t=1}^{T}(\hat{u}_t - x_t)^2. 
```
The process loss compares the model predictions to the estimated state varaibles. The predictions are calcualted by solving the differntial equation over the interval between observations
```math
u_{t+\Delta t} \approx  F(u_t,\Delta t;w,b) = u_t + \int_t^{t+\Delta t} NN(u;w,b)du,
```
where ``F`` is a function representing the soltuon to the differntial equation. The As before the loss is calcualted using the mean squared error 
```math
L_{process} = \frac{1}{T-1}\sum_{t=2}^{T}\left(\hat{u}_t - F(\hat{u}_{t-1},\Delta t; w,b)\right)^2. 
```
The full loss funcion is the weighted sum of the process and observation losses
```math
L = \omega_{observation}\times L_{observation}+\omega_{process}\times L_{process}.
```
The default training procedure will give equal weight to the process and observaitonal loss terms. However, if observaiton errors are large increasing the process model weight ``\omega_{process}`` can improve performance. Siuilarlly, if the dynamics of the system are highly variable over time, increasing the observation weight ``\omega_{observaiton}`` can help incentivise the model to recover these more complex dynamics. 

A regualrization term can also be added to the model to reduce over fitting. Users can choose between ``L1`` and ``L2`` regualrization terms. These terms add the magnitude of the neural network weights to the loss fucntion weighted by a factor ``\omega_{regualrization}``

```math
L = \omega_{observation}\times L_{observation}+\omega_{process}\times L_{process} + \omega_{regularization}||w||_{L_i}.
```

Simualted a data set from the Loka Volterra predator prey model.
```@example LVexample; continued = true
using Plots # hide
push!(LOAD_PATH,"../../src/")
using UniversalDiffEq
data,plt = LokaVolterra()
savefig(plt, "LVdata-plot.svg"); nothing # hide
```
![](LVdata-plot.svg)

Split test and training data 

```@example LVexample ; continued = true
Ntest = 20
training_data = data[1:(end - Ntest),:]
test_data = data[(end-Ntest):end,:]
training_data[1:4,:]
```

```@example LVexample ;continued = true
model = NODE(training_data;hidden_units=20,seed=123,proc_weight=2.0,obs_weight=1.0,reg_weight=10^-3.5,l = 0.5,extrap_rho = 0.0,reg_type = "L2")
gradient_decent!(model,step_size = 0.05,maxiter=2,verbos = false)
# BFGS!(model,verbos = false)
nothing # hide
```

```@example LVexample ;continued = true
plot_state_estiamtes(model)
savefig( "LVstates-plot.svg"); nothing # hide
```
![](LVstates-plot.svg)

```@example LVexample ;continued = true
plot_predictions(model)
savefig( "LVpreds-plot.svg"); nothing # hide
```
![](LVpreds-plot.svg)

```@example LVexample ;continued = true
plt,(p1,p2) = plot_forecast(model, test_data)
savefig(plt, "LVfcast-plot.svg"); nothing # hide
```
![](LVfcast-plot.svg)

```@example LVexample 
plot_predictions(model, test_data)
savefig("LVpredsout-plot.svg"); nothing # hide
```
![](LVpredsout-plot.svg)