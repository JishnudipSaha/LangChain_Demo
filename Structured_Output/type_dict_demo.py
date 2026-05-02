from typing import Type, TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {"name": "Jishnu", "age": 22}
print(new_person)