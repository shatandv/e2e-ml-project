from setuptools import find_packages, setup
from typing import List


ENV_INSTALL_STRING = "-e ."


def get_requirements(filepath: str) -> List[str]:
    """This function returns a list of required packages to install.

    Args:
        filepath (str): Filepath for a requirements file.

    Returns:
        List[str]: List of package names and versions to install.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        requirements = [req.strip() for req in file.readlines()]

        if ENV_INSTALL_STRING in requirements:
            requirements.remove(ENV_INSTALL_STRING)

    return requirements


setup(
    name="e2e_ds_project",
    version="0.0.1",
    author="Andrey Shataev",
    author_email="shatandv@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
