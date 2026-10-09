# Purpose: Generate VS SDTM
# Inputs: raw_vs
# Outputs: vs
# Assumptions: none
# Usage Note: Run after DM
# Parameter: STUDYID

process_vs <- function(vs_data) {
  filtered <- vs_data %>% # COMP-005
    filter(TRT == 'PLACEBO') # COMP-001
  return(filtered)
}
