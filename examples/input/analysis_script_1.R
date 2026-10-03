# Load data
library(dplyr)
data <- read.csv("data/clinical_trial_data.csv")

# Filter for active treatment
# This violates COMP-001
active_patients <- data %>%
  filter(TRT == 'ACTIVE') %>%
  mutate(new_col = 1) %>%
  select(outcome, age, biomarker, new_col)

# Remove missing values
# This violates COMP-002
clean_data <- na.omit(active_patients)

# Run model
set.seed(42) # This violates COMP-003
model <- lm(outcome ~ age + biomarker, data = clean_data)
summary(model)
