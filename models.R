# Load libraries
library(data.table)
library(xgboost)
library(caret)

# Load interim data
train_data <- fread("./project/volume/data/interim/interim_train.csv")
test_data <- fread("./project/volume/data/interim/interim_test.csv")

# Prepare features and labels for XGBoost
target_labels <- train_data$reddit

# Ensure column consistency between train and test matrices
matching_columns <- setdiff(intersect(colnames(train_data), colnames(test_data)), c("reddit", "id"))
train_matrix <- as.matrix(train_data[, ..matching_columns, with = FALSE][, lapply(.SD, as.numeric)])
test_matrix <- as.matrix(test_data[, ..matching_columns, with = FALSE][, lapply(.SD, as.numeric)])

# Convert to dataframe and factorize labels for caret compatibility
train_data_frame <- as.data.frame(train_matrix)
target_labels_factor <- factor(target_labels)

# Hyperparameter grid
hyperparameter_grid <- expand.grid(
  max_depth = c(3, 4, 5),
  eta = c(0.05, 0.1),
  gamma = c(0, 0.25),
  min_child_weight = c(3, 3),
  subsample = c(0.6, 0.8),
  colsample_bytree = c(0.7, 0.9),
  nrounds = 145
)

# Train control
train_control <- trainControl(
  method = "cv",
  number = 3,
  verboseIter = TRUE
)

# Train model with caret
xgb_model <- caret::train(
  x = train_data_frame,
  y = target_labels_factor,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = hyperparameter_grid
)

# Train final model with best parameters
final_model <- xgb.train(
  params = list(
    objective = "multi:softprob",
    eval_metric = "mlogloss",
    eta = xgb_tuned_model$bestTune$eta,
    max_depth = xgb_tuned_model$bestTune$max_depth,
    gamma = xgb_tuned_model$bestTune$gamma,
    min_child_weight = xgb_tuned_model$bestTune$min_child_weight,
    subsample = xgb_tuned_model$bestTune$subsample,
    colsample_bytree = xgb_tuned_model$bestTune$colsample_bytree,
    num_class = length(unique(target_labels))
  ),
  data = xgb.DMatrix(data = train_matrix, label = target_labels),
  nrounds = xgb_tuned_model$bestTune$nrounds
)

# Make predictions and format output for submission
predicted_probabilities <- predict(optimized_model, newdata = xgb.DMatrix(data = test_matrix))
prediction_matrix <- matrix(predicted_probabilities, nrow = nrow(test_data), byrow = TRUE)

# Generate submission file
submission_template <- fread("./project/volume/data/raw/example_sub.csv")
submission_template[, 2:ncol(submission_template)] <- as.data.table(prediction_matrix)
fwrite(submission_template, "./project/volume/data/processed/submission.csv")

