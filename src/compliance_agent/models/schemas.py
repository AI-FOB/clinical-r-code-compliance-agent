from pydantic import BaseModel, Field

class ComplianceFinding(BaseModel):
    """
    Structured representation of a compliance finding in R code.
    """
    rule_id: str = Field(
        ..., 
        description="Identifier of the applicable synthetic rule"
    )
    severity: str = Field(
        ..., 
        description="Potential severity of the finding"
    )
    finding: str = Field(
        ..., 
        description="Description of the identified issue"
    )
    evidence: str = Field(
        ..., 
        description="Relevant evidence from the R source code"
    )
    recommendation: str = Field(
        ..., 
        description="Suggested remediation"
    )
    confidence: float = Field(
        ..., 
        description="Model confidence in the finding",
        ge=0.0, 
        le=1.0
    )
