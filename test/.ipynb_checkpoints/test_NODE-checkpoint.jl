using UniversalDiffEq

#Test creating a NODE from a sample time series
sampleTimeSeries = rand(2,10)
testModel, testFit = createModel(sampleTimeSeries)

#Test creating a NODE from a file
testModel, testFit = createModel("randTest.csv")