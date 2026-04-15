from src.api.schemas import EmployeeInput
from pydantic import ValidationError


def validate_employee(data: dict) -> EmployeeInput:
    try:
        return EmployeeInput(**data)
    except ValidationError as e:
        raise ValueError(str(e))