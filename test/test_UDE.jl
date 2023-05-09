using UniversalDiffEq

knownDynamics(x,p,q) = [-p[1].*x[1].+q[1],0]

#Test creating a UDE from a sample time series
sampleTimeSeries = rand(2,10)
testModel, testFit = createModel(sampleTimeSeries,neededParameters=1,givenParameters=[1.])

#Test creating a UDE from a file
testModel, testFit = createModel("randTest.csv",neededParameters=1,givenParameters=[1.])