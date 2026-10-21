from pydantic import BaseModel,Field
from typing import Optional
class ResearchSummary(BaseModel):
    title: Optional[str]= Field(default=None,description="Title of the research paper, if available.")
    
    overview: str = Field(
        description="A concise overview of the document."
    )

    key_findings: list[str] = Field(
        description="The main findings and important observations."
    )

    methodology: str = Field(
        description="The methods or approaches described in the document."
    )

    limitations: list[str] = Field(
        description="Limitations explicitly identified in the document."
    )

    conclusion: str = Field(
        description="The main conclusion of the document."
    )