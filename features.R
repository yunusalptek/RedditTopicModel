# Load libraries
library(data.table)
library(Rtsne)
library(ggplot2)

# Set seed
set.seed(123)

# Load embeddings and metadata
kaggle_train <- fread("./project/volume/data/raw/kaggle_train.csv")
kaggle_test <- fread("./project/volume/data/raw/kaggle_test.csv")
train_emb <- fread("./project/volume/data/raw/train_emb.csv")
test_emb <- fread("./project/volume/data/raw/test_emb.csv")

# Encode target variable as integer for classification
kaggle_train$reddit <- as.integer(as.factor(kaggle_train$reddit)) - 1

# Combine embeddings and metadata
train_data <- cbind(train_emb, kaggle_train)
test_data <- cbind(test_emb, kaggle_test)

# Apply dimensionality reduction (PCA and t-SNE)
pca_model <- prcomp(rbind(train_emb, test_emb), center = TRUE, scale. = TRUE)
pca_train <- data.table(unclass(pca_model)$x[1:nrow(train_emb), ])
pca_test <- data.table(unclass(pca_model)$x[(nrow(train_emb) + 1):nrow(rbind(train_emb, test_emb)), ])

# Run t-SNE on PCA data
tsne_train <- Rtsne(as.matrix(pca_train), dims = 2, perplexity = 45, verbose = TRUE, max_iter = 600, check_duplicates = FALSE)
tsne_test <- Rtsne(as.matrix(pca_test), dims = 2, perplexity = 45, verbose = TRUE, max_iter = 600, check_duplicates = FALSE)

# Combine t-SNE results with data
train_tsne <- data.table(tsne_train$Y)
test_tsne <- data.table(tsne_test$Y)
train_data <- cbind(train_data, train_tsne)
test_data <- cbind(test_data, test_tsne)

# Save interim data
fwrite(train_data, "./project/volume/data/interim/interim_train.csv")
fwrite(test_data, "./project/volume/data/interim/interim_test.csv")