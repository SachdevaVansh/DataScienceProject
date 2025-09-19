from setuptools import setup, find_packages
from typing import List 

def get_requirements() -> List[str]:

    requirement_list=[]
    try:
        with open("requirements.txt") as f:
            lines=f.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!="-e .":
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found.")

    return requirement_list

#print(get_requirements())

setup(
    name="DataScienceProject",
    version="0.1.0",
    author="Vansh Sachdeva",
    packages=find_packages(),
    install_requires=get_requirements(),
    description="A Data Science Project",
)

                