from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Tashdid'
    age : Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0,lt=4, default=3.5, description="Result")


new_student = {
    'name' : 'Tashdid'
}

Student = Student(**new_student)

Student.model_dump()