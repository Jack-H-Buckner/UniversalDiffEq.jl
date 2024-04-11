using Revise, UniversalDiffEq, DataFrames

T = 20
data = DataFrame(t = 1:T,x = rand(T))
X = DataFrame(t = 1:T,x = rand(T))

model = NODE(data, X)

kfold_cv(model,k=3,leave_out=5)

T = 20
data = DataFrame(t = 1:T,x = rand(T))
X = DataFrame(t = 1:T,x = rand(T))

model = NODE(data)

kfold_cv(model,k=3,leave_out=5)



data = DataFrame(series =vcat(vcat(repeat([1],10),repeat([2],10)),repeat([3],10)) , t = vcat(1:10.0,1.0:20),x = rand(30))
X = DataFrame(series =vcat(vcat(repeat([1],10),repeat([2],10)),repeat([3],10)) , t = vcat(1:10.0,1.0:20),x = rand(30))

model = UniversalDiffEq.MultiNODE(data)

predicted_data,testing_data = kfold_cv(model,k=4,leave_out=2,maxiter=2)

model = UniversalDiffEq.MultiNODE(data,X)

predicted_data,testing_data = kfold_cv(model,k=4,leave_out=2,maxiter=2)
