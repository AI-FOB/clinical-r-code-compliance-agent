# Missing required header keywords (missing Purpose, Inputs, Outputs, etc.)
# This will trigger COMP-004 on line 1.

library(dplyr)

create_adsl <- function(dm_data) {
  set.seed(123) # COMP-003
  
  adsl <- dm_data %>% # COMP-005
    filter(TRT == 'ACTIVE') %>% # COMP-001, COMP-005
    mutate(flag = 1) # COMP-006
  
  return(adsl)
}
