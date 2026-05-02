import email

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Annotated


class Student(BaseModel):
    name: str = None
    age: Optional[int] = None
    email: Optional[EmailStr] = None
    cgpa: float = Field(ge=0, le=10)
    
new_std = {
    "name": "Jishnu",
    "age": '22',
    "email": "jishnudip@gmail.com",
    "cgpa": 9.5
    }

std = Student(**new_std)

student_dict = dict(std)

student_json = std.model_dump_json()

print(student_dict['age'])
print(student_json)