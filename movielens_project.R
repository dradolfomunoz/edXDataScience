##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table) # To prevent the script from halting, we had to install the data.table package

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Movie Rating Prediction using Machine Learning
##########################################################

# Install necessary package data.table if not already installed
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(magrittr)) install.packages("magrittr", repos = "http://cran.us.r-project.org")

# Load the required library
library(data.table)
library(magrittr)

# Handle missing values by replacing them with the mean of the columns
edx <- edx %>%
  mutate(rating = ifelse(is.na(rating), mean(rating, na.rm = TRUE), rating),
         userId = ifelse(is.na(userId), mean(userId, na.rm = TRUE), userId),
         movieId = ifelse(is.na(movieId), mean(movieId, na.rm = TRUE), movieId))

# Check for remaining missing values
if(any(is.na(edx))) {
  cat("There are remaining missing values in the data. Please investigate.\n")
} else {
  cat("No missing values found after normalization.\n")
}

# Further split the edx dataset into training and testing sets for model development
set.seed(1, sample.kind = "Rounding")
train_index <- createDataPartition(y = edx$rating, times = 1, p = 0.9, list = FALSE)
train_set <- edx[train_index,]
test_set <- edx[-train_index,]

# Ensure userId and movieId in the test set are also in the train set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Baseline Model: Predict the average movie rating
mu_hat <- mean(train_set$rating)

# Compute RMSE for the baseline model
rmse_baseline <- sqrt(mean((test_set$rating - mu_hat)^2))
cat("Baseline RMSE:", rmse_baseline, "\n")

# Advanced Model: Regularized Movie and User Effects Model

# 1. Calculate overall average rating
mu <- mean(train_set$rating)

# 2. Regularization parameter (lambda) - determine through cross-validation
lambdas <- seq(0, 10, 0.25)
calculate_rmse <- function(lambda) {
  # Compute movie effects
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  # Compute user effects
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))
  
  # Predict ratings based on movie and user effects
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  # Calculate RMSE
  return(sqrt(mean((test_set$rating - predicted_ratings)^2)))
}

# Find optimal lambda
rmse_results <- sapply(lambdas, calculate_rmse)
best_lambda <- lambdas[which.min(rmse_results)]
best_rmse <- min(rmse_results)

cat("Optimal lambda:", best_lambda, "\n")
cat("Best RMSE (Cross-Validation):", best_rmse, "\n")

# Final Model: Train on the entire edx dataset and evaluate on final_holdout_test

# Compute movie and user effects using the optimal lambda
b_i_final <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + best_lambda))

b_u_final <- edx %>%
  left_join(b_i_final, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + best_lambda))

# Predict ratings for final_holdout_test
predicted_final <- final_holdout_test %>%
  left_join(b_i_final, by = "movieId") %>%
  left_join(b_u_final, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Compute RMSE on the final_holdout_test set
rmse_final <- sqrt(mean((final_holdout_test$rating - predicted_final)^2))
cat("Final RMSE on final hold-out test set:", rmse_final, "\n")

