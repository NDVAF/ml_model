from pydantic import BaseModel, validator


class DecisionTreeClassifierParams(BaseModel):
    criterion: str = "entropy"
    max_depth: int = 8
    min_samples_leaf: int = 10
    random_state: int = 42

    @validator("criterion")
    def validate_criterion(cls, value):
      allowed_criterions = ["gini", "entropy"]
      if value not in allowed_criterions:
           raise ValueError(f"criterion must be one of {allowed_criterions}")
      return value

    @validator("max_depth")
    def validate_max_depth(cls, value):
      if value <= 0:
         raise ValueError("max_depth must be a positive integer")
      return value
    
    @validator("min_samples_leaf")
    def validate_min_samples_leaf(cls, value):
         if value <= 0:
            raise ValueError("min_samples_leaf must be a positive integer")
         return value
    
    @validator("random_state")
    def validate_random_state(cls, value):
         if value < 0:
           raise ValueError("random_state must be a non negative integer")
         return value