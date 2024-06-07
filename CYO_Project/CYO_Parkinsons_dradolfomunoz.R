##############################################################
###'CYO_Project:Capstone -  PH125.9x HarvardX Data Science'###
###author: "Adolfo Muñoz Macho @dradolfomunoz"################
##################date: "2024-06-07"##########################

# **Parkinson's Disease Detection Using Voice Measurements.**

## 1. Introduction.

### 1.1 Dataset Information.

### 1.2 Variables Information.

### 1.3 Objetives.

## 2. Methods and Analysis.

### 2.1 Downloading and Preprocessing the Dataset.

# Ensure necessary packages are installed and loaded
if (!require(tidyverse)) install.packages("tidyverse", dependencies = TRUE)
if (!require(caret)) install.packages("caret", dependencies = TRUE)
if (!require(e1071)) install.packages("e1071", dependencies = TRUE)
if (!require(treemap)) install.packages("treemap", dependencies = TRUE)
if (!require(knitr)) install.packages("knitr", dependencies = TRUE)
if (!require(kableExtra)) install.packages("kableExtra", dependencies = TRUE)

library(tidyverse)
library(caret)
library(e1071)
library(treemap)
library(knitr)
library(kableExtra)

#### 2.1.1 Download the Dataset.
                                                                                                                                            
#### 2.1.2 Load the Dataset.


#### 2.1.3 Initial Inspection.


#### 2.1.4 Data Visualization.
                                                                                                                                          

# 2. Download the Dataset
 # URL of the dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

# Path to save the dataset
data_path <- "parkinsons.data"

# Download the dataset if it does not exist
if (!file.exists(data_path)) {
  download.file(url, data_path)
}

# Load the dataset
parkinsons_data <- read_csv(data_path)

# Test for missing values
if (sum(is.na(parkinsons_data)) == 0) {
  message("The dataset does not have any missing values.")
}

# Display a sample of the dataset (6 rows and 6 columns)
kable(head(parkinsons_data[, 1:6]), caption = "Sample of the Dataset (6 rows and 6 columns)")


# 4. Present the Distribution of Data with a Histogram
# Plot histogram for a specific column (e.g., average fundamental frequency)
ggplot(parkinsons_data, aes(x = `MDVP:Fo(Hz)`)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "black") +
  labs(title = "Distribution of Average Fundamental Frequency (Fo)", x = "Average Fundamental Frequency (Hz)", y = "Count")

# Calculate mean values for different features grouped by health status
  mean_values <- parkinsons_data %>%
   group_by(status) %>%
   summarize(across(where(is.numeric), mean, na.rm = TRUE)) %>%
   pivot_longer(-status, names_to = "Feature", values_to = "Mean")
                                                                          
# Plot mean values for different features grouped by health status
 ggplot(mean_values, aes(x = Feature, y = Mean, fill = as.factor(status))) + geom_col(position = "dodge") + labs(title = "Mean Distribution of Features by Health Status", x = "Feature", y = "Mean") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_fill_manual(values = c("0" = "green", "1" = "red"), name = "Health Status", labels = c("Healthy", "Parkinson's"))
                                                                        
                                                                          


# Select some representative features
selected_features <- parkinsons_data %>%
  select(status, `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`, `HNR`)

# Calculate the mean of these features grouped by health status
feature_means <- selected_features %>%
  group_by(status) %>%
  summarize(across(everything(), mean, na.rm = TRUE))

# Melt the data for plotting
feature_means_long <- feature_means %>%
  pivot_longer(-status, names_to = "Feature", values_to = "Mean")

feature_means_long$status <- ifelse(feature_means_long$status == 0, "Healthy", "Parkinson's")

# Create a treemap
treemap(feature_means_long,
        index = c("status", "Feature"),
        vSize = "Mean",
        title = "Treemap of Features by Health Status",
        palette = c("Healthy" = "#00FF00", "Parkinson's" = "#FF0000"),
                                                                          border.col = "white")

# Create a scatter plot for two variables (e.g., `MDVP:Fo(Hz)` vs `HNR`)
ggplot(parkinsons_data, aes(x = `MDVP:Fo(Hz)`, y = HNR, color = as.factor(status))) +
  geom_point(alpha = 0.6) +
  labs(title = "Scatter Plot of Fo(Hz) vs HNR", x = "Average Fundamental Frequency (Fo)", y = "HNR") +
  scale_color_manual(values = c("0" = "#00FF00", "1" = "#FF0000"), name = "Health Status", labels = c("Healthy", "Parkinson's"))

 # Create a violin plot to compare the distribution of `HNR` between health statuses
ggplot(parkinsons_data, aes(x = as.factor(status), y = HNR, fill = as.factor(status))) +
  geom_violin() +
  labs(title = "Violin Plot of HNR by Health Status", x = "Health Status", y = "HNR") +
  scale_fill_manual(values = c("0" = "#00FF00", "1" = "#FF0000"), name = "Health Status", labels = c("Healthy", "Parkinson's"))

# Create a line plot to observe the trends of `MDVP:Fo(Hz)` through records
ggplot(parkinsons_data, aes(x = 1:nrow(parkinsons_data), y = `MDVP:Fo(Hz)`, color = as.factor(status))) +
  geom_line() +
  labs(title = "Line Plot of MDVP:Fo(Hz) through Records", x = "Record Index", y = "Average Fundamental Frequency (Fo)") +
  scale_color_manual(values = c("0" = "#00FF00", "1" = "#FF0000"), name = "Health Status", labels = c("Healthy", "Parkinson's"))

# Create a box plot for `MDVP:Fo(Hz)` segmented by health status
ggplot(parkinsons_data, aes(x = as.factor(status), y = `MDVP:Fo(Hz)`, fill = as.factor(status))) +
  geom_boxplot() +
  labs(title = "Box Plot of MDVP:Fo(Hz) by Health Status", x = "Health Status", y = "Average Fundamental Frequency (Fo)") +
  scale_fill_manual(values = c("0" = "#00FF00", "1" = "#FF0000"), name = "Health Status", labels = c("Healthy", "Parkinson's"))

# Create a density plot for `MDVP:Fo(Hz)` segmented by health status
ggplot(parkinsons_data, aes(x = `MDVP:Fo(Hz)`, fill = as.factor(status))) +
  geom_density(alpha = 0.6) +
  labs(title = "Density Plot of MDVP:Fo(Hz) by Health Status", x = "Average Fundamental Frequency (Fo)", y = "Density") +
  scale_fill_manual(values = c("0" = "#00FF00", "1" = "#FF0000"), name = "Health Status", labels = c("Healthy", "Parkinson's"))


### 2.2 Modeling Approach.

#### 2.2.1 Selection of Machine Learning Algorithms.

### 2.3. Data Preparation:

#### 2.3.1. Dataset Splitting.

#### 2.3.2. Model Training.LR,SVM,RF,KNN

#### 2.3.3. Model Evaluation.

# Make sure that the necessary packages are installed and loaded
if (!require(tidyverse)) install.packages("tidyverse", dependencies = TRUE)
if (!require(caret)) install.packages("caret", dependencies = TRUE)
if (!require(e1071)) install.packages("e1071", dependencies = TRUE)
if (!require(treemap)) install.packages("treemap", dependencies = TRUE)
if (!require(knitr)) install.packages("knitr", dependencies = TRUE)

library(tidyverse)
library(caret)
library(e1071)
library(treemap)
library(knitr)

# Dataset URL
url <- "https://archive.ics.uci.edu/static/public/174/parkinsons.zip"

# Save the zip file path
zip_path <- "parkinsons.zip"
data_path <- "parkinsons.data"

# Download the zip file if it does not exist
if (!file.exists(zip_path)) {
  download.file(url, zip_path)
}

# Unzip the zip file if the file 'parkinsons.data' does not exist
if (!file.exists(data_path)) {
  unzip(zip_path)
}

# Read the dataset
parkinsons_data <- read.csv(data_path)

# Display a sample of the dataset to ensure it has been loaded correctly
head(parkinsons_data)

# Display dataset structure
str(parkinsons_data)

# Check for missing values
sum(is.na(parkinsons_data))

# Remove the 'name' column as it is not relevant for the model.
parkinsons_data <- parkinsons_data %>% select(-name)

#Preprocessing the dataset 
#Convert the target variable 'status' to a factor
parkinsons_data$status <- as.factor(parkinsons_data$status)

# Split the data into training and testing sets (80/20 ratio)
set.seed(123)
trainIndex <- createDataPartition(parkinsons_data$status, p = 0.8, list = FALSE)
train_data <- parkinsons_data[trainIndex, ]
test_data <- parkinsons_data[-trainIndex, ]

# Confirm the division
table(train_data$status)
table(test_data$status)

# Logistic Regression
model_logistic <- train(status ~ ., data = train_data, method = "glm", family = binomial)
predictions_logistic <- predict(model_logistic, newdata = test_data)
conf_matrix_logistic <- confusionMatrix(predictions_logistic, test_data$status)

print(model_logistic)
print(conf_matrix_logistic)

# Support Vector Machines
model_svm <- train(status ~ ., data = train_data, method = "svmRadial")
predictions_svm <- predict(model_svm, newdata = test_data)
conf_matrix_svm <- confusionMatrix(predictions_svm, test_data$status)

print(model_svm)
print(conf_matrix_svm)

# Random Forest
model_rf <- train(status ~ ., data = train_data, method = "rf")
predictions_rf <- predict(model_rf, newdata = test_data)
conf_matrix_rf <- confusionMatrix(predictions_rf, test_data$status)

print(model_rf)
print(conf_matrix_rf)

# K-Nearest Neighbors
model_knn <- train(status ~ ., data = train_data, method = "knn", tuneLength = 10)
predictions_knn <- predict(model_knn, newdata = test_data)
conf_matrix_knn <- confusionMatrix(predictions_knn, test_data$status)

print(model_knn)
print(conf_matrix_knn)

# Compare models
resamples <- resamples(list(Logistic=model_logistic, SVM=model_svm, RandomForest=model_rf, KNN=model_knn))
summary(resamples)
dotplot(resamples)

## 3. Results.Below are the key performance metrics obtained for each of the four machine learning models: Logistic Regression, Support Vector Machine (SVM), Random Forest, and K-Nearest Neighbors (KNN).


# Create a metric table
metrics_table <- data.frame(
  Model = c("Logistic Regression", "SVM", "Random Forest", "KNN"),
  Accuracy = c(conf_matrix_logistic$overall['Accuracy'],
               conf_matrix_svm$overall['Accuracy'],
               conf_matrix_rf$overall['Accuracy'],
               conf_matrix_knn$overall['Accuracy']),
  Sensitivity = c(conf_matrix_logistic$byClass['Sensitivity'],
                  conf_matrix_svm$byClass['Sensitivity'],
                  conf_matrix_rf$byClass['Sensitivity'],
                  conf_matrix_knn$byClass['Sensitivity']),
  Specificity = c(conf_matrix_logistic$byClass['Specificity'],
                  conf_matrix_svm$byClass['Specificity'],
                  conf_matrix_rf$byClass['Specificity'],
                  conf_matrix_knn$byClass['Specificity'])
)

# Show Table
kable(metrics_table, caption = "Model Comparison")

# Convert table to long format for ggplot2

metrics_long <- metrics_table %>%
  pivot_longer(cols = c("Accuracy", "Sensitivity", "Specificity"), names_to = "Metric", values_to = "Value")

# Create a pastel color palette
pastel_colors <- c("#FBB4AE", "#B3CDE3", "#CCEBC5", "#DECBE4")

# Plot the metrics using pastel colors
ggplot(metrics_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Comparison of Model Metrics",
       x = "Metric",
       y = "Value",
       fill = "Model") +
  scale_fill_manual(values = pastel_colors) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),  # Center and enhance the title
    axis.text = element_text(size = 12),                              # Improve axis text size
    axis.title = element_text(size = 14),                             # Improve axis title size
    legend.title = element_text(size = 12),                           # Improve legend title size
    legend.text = element_text(size = 10)    
  )

