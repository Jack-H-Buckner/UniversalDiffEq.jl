# Examples

## NODE
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
gradient_decent!(model,step_size = 0.05,maxiter=250)
BFGS!(model,verbos = true)
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
plt,(p1,p2) = UniversalDiffEq.plot_forecast(model, test_data)
savefig(plt, "LVfcast-plot.svg"); nothing # hide
```
![](LVfcast-plot.svg)

```@example LVexample 
plt,(p1,p2) = plot_predictions(model, test_data)
savefig(plt, "LVpredsout-plot.svg"); nothing # hide
```
![](LVpredsout-plot.svg)