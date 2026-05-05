############################################################
# Clinical Prediction Modelling Practice Project
# Predicting Motor Outcome Using Simulated DBS-Style Data
#
# This project uses simulated data only.
# It does not represent real DBS patients or real clinical findings.
############################################################

# -----------------------------
# 1. Load packages
# -----------------------------

# Install packages if needed:
# install.packages(c("tidyverse", "glmnet", "broom"))

library(tidyverse)
library(glmnet)
library(broom)

# -----------------------------
# 2. Set seed for reproducibility
# -----------------------------

set.seed(123)

# -----------------------------
# 3. Simulate DBS-style dataset
# -----------------------------

# Number of simulated participants
n <- 150

sim_data <- tibble(
  id = 1:n,
  
  # Clinical variables
  age = round(rnorm(n, mean = 62, sd = 8), 1),
  sex = sample(c("Male", "Female"), size = n, replace = TRUE, prob = c(0.6, 0.4)),
  disease_duration = round(rnorm(n, mean = 10, sd = 4), 1),
  baseline_updrs_iii = round(rnorm(n, mean = 45, sd = 12), 1),
  levodopa_response_pct = round(rnorm(n, mean = 45, sd = 15), 1),
  
  # Neuropsychological and mood variables
  moca_score = round(rnorm(n, mean = 26, sd = 3), 1),
  depression_anxiety_score = round(rnorm(n, mean = 10, sd = 5), 1),
  
  # Imaging-style proxy variables
  stn_connectivity_score = round(rnorm(n, mean = 0, sd = 1), 2),
  electrode_distance_proxy = round(rnorm(n, mean = 2.5, sd = 1), 2)
)

# Keep values within plausible ranges
sim_data <- sim_data %>%
  mutate(
    age = pmin(pmax(age, 40), 80),
    disease_duration = pmin(pmax(disease_duration, 2), 25),
    baseline_updrs_iii = pmin(pmax(baseline_updrs_iii, 15), 80),
    levodopa_response_pct = pmin(pmax(levodopa_response_pct, 5), 85),
    moca_score = pmin(pmax(moca_score, 15), 30),
    depression_anxiety_score = pmin(pmax(depression_anxiety_score, 0), 30),
    electrode_distance_proxy = pmin(pmax(electrode_distance_proxy, 0.2), 6)
  )

# -----------------------------
# 4. Simulate outcome
# -----------------------------

# The simulated outcome is based on plausible assumptions:
# better levodopa response, stronger STN connectivity, and lower electrode distance
# are associated with greater motor improvement.
#
# These are simulated relationships and should not be interpreted as real DBS evidence.

sim_data <- sim_data %>%
  mutate(
    motor_improvement_pct =
      20 +
      0.35 * levodopa_response_pct +
      0.15 * baseline_updrs_iii -
      0.20 * age -
      0.30 * disease_duration +
      1.20 * moca_score -
      0.40 * depression_anxiety_score +
      4.00 * stn_connectivity_score -
      3.50 * electrode_distance_proxy +
      rnorm(n, mean = 0, sd = 8)
  ) %>%
  mutate(
    motor_improvement_pct = pmin(pmax(motor_improvement_pct, 0), 80),
    responder = ifelse(motor_improvement_pct >= 30, 1, 0)
  )

# -----------------------------
# 5. Add small amount of missing data
# -----------------------------

# This mimics the reality that clinical datasets often contain some missing values.

set.seed(456)

add_missing <- function(x, prop_missing = 0.05) {
  missing_index <- sample(1:length(x), size = floor(prop_missing * length(x)))
  x[missing_index] <- NA
  return(x)
}

sim_data <- sim_data %>%
  mutate(
    moca_score = add_missing(moca_score, 0.05),
    depression_anxiety_score = add_missing(depression_anxiety_score, 0.05),
    stn_connectivity_score = add_missing(stn_connectivity_score, 0.05)
  )

# -----------------------------
# 6. Save simulated dataset
# -----------------------------

# Create data folder if it does not exist
if (!dir.exists("data")) {
  dir.create("data")
}

write_csv(sim_data, "data/simulated_dbs_data.csv")

# -----------------------------
# 7. Inspect data
# -----------------------------

# View first few rows
head(sim_data)

# Check structure
glimpse(sim_data)

# Summary statistics
summary(sim_data)

# -----------------------------
# 8. Inspect missing data
# -----------------------------

missing_summary <- sim_data %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "missing_count"
  ) %>%
  mutate(
    missing_percent = round((missing_count / nrow(sim_data)) * 100, 1)
  )

print(missing_summary)

# Simple missing data plot
missing_summary %>%
  ggplot(aes(x = reorder(variable, missing_percent), y = missing_percent)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Percentage of Missing Data by Variable",
    x = "Variable",
    y = "Missing data (%)"
  )

# -----------------------------
# 9. Simple complete-case dataset
# -----------------------------

# For this beginner project, we use complete-case analysis.
# In real clinical research, multiple imputation may be more appropriate.

model_data <- sim_data %>%
  drop_na()

# Check sample size after removing missing values
nrow(model_data)

# -----------------------------
# 10. Descriptive summary
# -----------------------------

descriptive_summary <- model_data %>%
  summarise(
    mean_age = mean(age),
    mean_disease_duration = mean(disease_duration),
    mean_baseline_updrs = mean(baseline_updrs_iii),
    mean_levodopa_response = mean(levodopa_response_pct),
    mean_moca = mean(moca_score),
    mean_motor_improvement = mean(motor_improvement_pct),
    responder_rate = mean(responder)
  )

print(descriptive_summary)

# -----------------------------
# 11. Train-test split
# -----------------------------

set.seed(789)

train_index <- sample(
  1:nrow(model_data),
  size = 0.75 * nrow(model_data)
)

train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# -----------------------------
# 12. Baseline linear regression model
# -----------------------------

baseline_lm <- lm(
  motor_improvement_pct ~ age +
    sex +
    disease_duration +
    baseline_updrs_iii +
    levodopa_response_pct +
    moca_score +
    depression_anxiety_score +
    stn_connectivity_score +
    electrode_distance_proxy,
  data = train_data
)

summary(baseline_lm)

# Tidy coefficient table
lm_coefficients <- tidy(baseline_lm)
print(lm_coefficients)

# -----------------------------
# 13. Evaluate baseline model
# -----------------------------

lm_predictions <- predict(baseline_lm, newdata = test_data)

lm_results <- tibble(
  observed = test_data$motor_improvement_pct,
  predicted = lm_predictions,
  residual = observed - predicted
)

lm_rmse <- sqrt(mean((lm_results$residual)^2))
lm_mae <- mean(abs(lm_results$residual))
lm_r_squared <- cor(lm_results$observed, lm_results$predicted)^2

lm_performance <- tibble(
  model = "Baseline linear regression",
  RMSE = lm_rmse,
  MAE = lm_mae,
  R_squared = lm_r_squared
)

print(lm_performance)

# -----------------------------
# 14. Plot predicted vs observed values
# -----------------------------

if (!dir.exists("outputs")) {
  dir.create("outputs")
}

predicted_plot <- lm_results %>%
  ggplot(aes(x = observed, y = predicted)) +
  geom_point(alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Predicted vs Observed Motor Improvement",
    subtitle = "Baseline linear regression model",
    x = "Observed motor improvement (%)",
    y = "Predicted motor improvement (%)"
  )

print(predicted_plot)

ggsave(
  filename = "outputs/predicted_vs_observed.png",
  plot = predicted_plot,
  width = 7,
  height = 5,
  dpi = 300
)

# -----------------------------
# 15. Penalized regression using LASSO
# -----------------------------

# glmnet requires a model matrix for predictors and a numeric outcome.

x_train <- model.matrix(
  motor_improvement_pct ~ age +
    sex +
    disease_duration +
    baseline_updrs_iii +
    levodopa_response_pct +
    moca_score +
    depression_anxiety_score +
    stn_connectivity_score +
    electrode_distance_proxy,
  data = train_data
)[, -1]

y_train <- train_data$motor_improvement_pct

x_test <- model.matrix(
  motor_improvement_pct ~ age +
    sex +
    disease_duration +
    baseline_updrs_iii +
    levodopa_response_pct +
    moca_score +
    depression_anxiety_score +
    stn_connectivity_score +
    electrode_distance_proxy,
  data = test_data
)[, -1]

y_test <- test_data$motor_improvement_pct

# Run 10-fold cross-validated LASSO
set.seed(101)

cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  nfolds = 10,
  standardize = TRUE
)

# Plot cross-validation curve
png("outputs/cv_lasso_plot.png", width = 800, height = 600)
plot(cv_lasso)
dev.off()

# Best lambda values
cv_lasso$lambda.min
cv_lasso$lambda.1se

# Fit final LASSO model using lambda.1se
# lambda.1se is often preferred for a simpler model.

lasso_predictions <- predict(
  cv_lasso,
  s = "lambda.1se",
  newx = x_test
)

lasso_results <- tibble(
  observed = y_test,
  predicted = as.numeric(lasso_predictions),
  residual = observed - predicted
)

lasso_rmse <- sqrt(mean((lasso_results$residual)^2))
lasso_mae <- mean(abs(lasso_results$residual))
lasso_r_squared <- cor(lasso_results$observed, lasso_results$predicted)^2

lasso_performance <- tibble(
  model = "LASSO regression",
  RMSE = lasso_rmse,
  MAE = lasso_mae,
  R_squared = lasso_r_squared
)

print(lasso_performance)

# Compare model performance
performance_comparison <- bind_rows(
  lm_performance,
  lasso_performance
)

print(performance_comparison)

# -----------------------------
# 16. Extract LASSO coefficients
# -----------------------------

lasso_coefficients <- coef(cv_lasso, s = "lambda.1se")

lasso_coefficients_df <- as.matrix(lasso_coefficients) %>%
  as.data.frame() %>%
  rownames_to_column(var = "variable") %>%
  rename(coefficient = s1) %>%
  filter(coefficient != 0)

print(lasso_coefficients_df)

# Plot non-zero LASSO coefficients
lasso_coef_plot <- lasso_coefficients_df %>%
  filter(variable != "(Intercept)") %>%
  ggplot(aes(x = reorder(variable, coefficient), y = coefficient)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Non-zero LASSO Coefficients",
    subtitle = "Using lambda.1se",
    x = "Variable",
    y = "Coefficient"
  )

print(lasso_coef_plot)

ggsave(
  filename = "outputs/lasso_coefficients.png",
  plot = lasso_coef_plot,
  width = 7,
  height = 5,
  dpi = 300
)

# -----------------------------
# 17. Cautious interpretation
# -----------------------------

# Important:
# These coefficients come from simulated data.
# They should not be interpreted as evidence about real DBS outcomes.
# The purpose is to demonstrate a reproducible modelling workflow.

cat("\nCautious interpretation:\n")
cat("This simulated workflow suggests how clinical, cognitive, mood, and imaging-style variables can be incorporated into interpretable prediction models.\n")
cat("Because the data are simulated, the results are not clinically meaningful and should not be used to draw conclusions about real DBS patients.\n")
cat("The value of the project is methodological: data simulation, preprocessing, regression modelling, penalized regression, cross-validation, performance evaluation, and transparent reporting.\n")
