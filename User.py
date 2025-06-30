from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., description="The name of the user")
    age: int = Field(gt=0, lt=100, description="The age of the user")