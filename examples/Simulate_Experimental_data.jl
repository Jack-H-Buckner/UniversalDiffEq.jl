using DataFrames, CSV, Random, Distributions, Plots
 
# derivatives of competing speceis system

function mortality(X1,X2,X3,p)
    m1 = exp(p.beta11*X1 + p.beta12*X2 + p.beta13*X3 + p.beta10)
    m2 = exp(p.beta21*X1 + p.beta22*X2 + p.beta23*X3 + p.beta20)
    return [m1,m2]
end 

function derivs(u,p,treatment)
    p1,p2,X1,X2,X3 = u
    E = 1- p1-p2
    m = mortality(X1,X2,X3,p)
    dp1 = (p.r1+p.alpha1*p1)*E - (1-treatment.exclusion)*p.m1*p1/(1+p.h*p1^3) - m[1]*p1 - treatment.removal1*p1
    dp2 = (p.r2+p.alpha2*p2)*E -(1-treatment.exclusion)*p.m2*p2 - m[2]*p2 - treatment.removal2*p2
    dX1 = p.rho*X1
    dX2 = p.rho*X2
    dX3 = p.rho*X3

    if p1 < 0
        dp1 = 0.1
    end 

    if p2 < 0
        dp2 = 0.1
    end 

    return [dp1,dp2,dX1,dX2,dX3]
end 

# set paramters 
parameters = (r1 = 0.1, r2 = 0.01, alpha1 =0.15, alpha2 = 0.5,  m1 = 0.05, m2 = 0.025, h = 10.0, 
                beta11 = 0.5, beta21 = 0.0, beta12 = 0.0, beta22 = 0.5, beta13 = 0.25, beta23 = 0.25,
                 beta10 = -3.0, beta20 = -3.0, rho = -0.1)
treatment1 = (exclusion = 0, removal1 = 0, removal2 = 0)
treatment2 = (exclusion = 1, removal1 = 0, removal2 = 0)
treatment3 = (exclusion = 0, removal1 = 1, removal2 = 0)
treatment4 = (exclusion = 0, removal1 = 0, removal2 = 1)

# simualtions
function simulate(T, dt, u0,parameters,treatment)
    N = length(collect(0:dt:T))
    data = DataFrame(t = 0:dt:T, p1 = zeros(N), p2 = zeros(N))
    X = DataFrame(t = 0:dt:T, X1 = zeros(N), X2 = zeros(N))
    u = u0; i = 0
    for t in 0:dt:T
        i+=1
        u .+= dt*derivs(u,parameters,treatment) .+ vcat(zeros(2),rand( Distributions.Normal(0,0.1),3))
        data.p1[i] = u[1];data.p2[i] = u[2]
        X.X1[i] = u[3]; X.X2[i] = u[4]
    end
    return data, X
end 

dt = 1200
Ntest = 10
data,X = simulate(1000, 0.01,[0.5,0.05,0.0,0.0,2.0],parameters,treatment1)
data1 = data[1:dt:(end-dt*Ntest),:]
testdata1 = data[(end-dt*Ntest):dt:end,:]
X1 = X[1:dt:end,:]
testdata1.series  .= 1
data1.series .= 1
X1.series .= 1

data,X = simulate(1000, 0.01,[0.5,0.05,0.0,0.0,2.0],parameters,treatment2)
data2 = data[1:dt:(end-dt*Ntest),:]
testdata2 = data[(end-dt*Ntest):dt:end,:]
X2 = X[1:dt:end,:]
testdata2.series .= 2
data2.series .= 2
X2.series .= 2

data,X = simulate(1000, 0.01,[0.5,0.05,0.0,0.0,2.0],parameters,treatment3)
data3 = data[1:dt:(end-dt*Ntest),:]
testdata3 = data[(end-dt*Ntest):dt:end,:]
X3 = X[1:dt:end,:]
testdata3.series .= 3
data3.series .= 3
X3.series .= 3

data,X = simulate(1000, 0.01,[0.5,0.05,0.0,0.0,2.0],parameters,treatment4)
data4 = data[1:dt:(end-dt*Ntest),:]
testdata4 = data[(end-dt*Ntest):dt:end,:]
X4 = X[1:dt:end,:]
testdata4.series .= 4
data4.series .= 4
X4.series .= 4


data = vcat(data1,data2)
data = vcat(data,data3)
data = vcat(data,data4)
data.t .= data.t./dt

testdata = vcat(testdata1,testdata2)
testdata = vcat(testdata,testdata3)
testdata = vcat(testdata,testdata4)
testdata.t .= testdata.t./dt

X = vcat(X1,X2)
X = vcat(X,X3)
X = vcat(X,X4)
X.t .= X.t ./dt
CSV.write("examples/experiments_trainig_data.csv",data)
CSV.write("examples/experiments_test_data.csv",testdata)
CSV.write("examples/experiments_covars.csv",X)