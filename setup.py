from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__)

long_description = "package to get topquartile returns given a set of stocks"
readme_path = HERE / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

requirements = []
req_path = HERE / "requirements.txt"
if req_path.exists():
    requirements = req_path.read_text().splitlines()

setup(
    name="your_package_name",
    version="0.1.0",
    author="G Money",
    author_email="g_money_the_pimp@email.com",
    description="package to pick stocks with top quartile returns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/osamkabinladna/topquartile",

    packages=find_packages(exclude=["tests", "docs"]),

    python_requires=">=3.11",

    install_requires=requirements,

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)