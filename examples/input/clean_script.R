# Purpose: Clean test script
# Inputs: data_path
# Outputs: summary
# Assumptions: None
# Usage Note: Run via source
# Parameter: trt_param

# Load data
library(dplyr)
library(tidyr)

# Assume seed is set in a global runner script
# Assume treatment mapping is passed as a parameter
run_analysis <- function(data_path, trt_param) {
  data <- read.csv(data_path)

  # Filter for active treatment dynamically
  active_patients <- data |>
    filter(TRT_CODE == trt_param)

  # Explicit missing value handling
  clean_data <- active_patients |>
    drop_na(outcome, age, biomarker)

  # Run model
  model <- lm(outcome ~ age + biomarker, data = clean_data)
  return(summary(model))
}
