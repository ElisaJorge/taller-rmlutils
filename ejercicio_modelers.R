library(RMLutils)
library(data.table)


data.dt <- generateRegressionData(1000, generate_random_weights = FALSE)
data.keys <- 1:nrow(data.dt)
smplr <- srsSampler(key = data.keys, ssize = 700)
train.keys <- getSample(smplr)
test.keys <- setdiff(data.keys, train.keys)

target <- "regression_target"
regressors <- setdiff(colnames(data.dt), target)
params <- list(
  "learning_rate" = 0.07,
  "nrounds" = 10,
  "objective" = "regression",
  "verbosity" = -1
)

modeler <- LGBRegressorModeler(params, target, regressors)

modeler <- train(modeler, data = data.dt, train.keys)

evaluate(modeler, data = data.dt, cost_function = MSECostFunction(), subSet = test.keys)
evaluate(modeler, data = data.dt, cost_function = MSECostFunction(), subSet = train.keys)
