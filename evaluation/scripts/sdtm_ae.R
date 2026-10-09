# Purpose: Generate AE SDTM
# Inputs: raw_ae
# Outputs: ae
# Assumptions: none
# Usage Note: Run after DM
# Parameter: STUDYID

clean_ae <- function(raw_ae) {
  # Drop missing values
  clean_data <- na.omit(raw_ae) # COMP-002
  
  clean_data |> 
    select(STUDYID, USUBJID, AETERM) # COMP-006
}
