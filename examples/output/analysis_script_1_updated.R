# Purpose: [Describe the script's overall goal.]
# Inputs: [Specify required input data and structure.]
# Outputs: [Specify the data product/result generated.]
# Assumptions: [List any underlying assumptions made during modeling.]
# Usage Note: [Provide instructions on executing the code.]
# Parameter: [Define any custom parameters used in the script.]
# Load data
library(dplyr)
data <- read.csv("data/clinical_trial_data.csv")

# Filter for active treatment
# This violates COMP-001
active_patients <- data |>
  filter(TRT == target_treatment_param) |>
  dplyr::mutate(new_col = 1) |>
  dplyr::select(outcome, age, biomarker, new_col)

# Remove missing values
# This violates COMP-002
clean_data <- active_patients |> tidyr::drop_na()

# Run model
set.seed(42) # This violates COMP-003
model <- lm(outcome ~ age + biomarker, data = clean_data)
summary(model)
