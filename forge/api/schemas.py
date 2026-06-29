from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    name: str = "Untitled Experiment"
    target_column: str
    task_description: str = ""
    trials: int = Field(default=10, ge=1, le=100)
    fast_mode: bool = True


class ExperimentResponse(BaseModel):
    id: str
    name: str
    target_column: str
    task_description: str
    status: str
    created_at: str
    progress: str
    error: str = ""
    result: dict = {}


class ExperimentStatusResponse(BaseModel):
    id: str
    status: str
    progress: str
    error: str = ""
