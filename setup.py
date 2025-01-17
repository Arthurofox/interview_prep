from setuptools import setup, find_packages

setup(
    name="interview_prep",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.4.2",
        "pydantic-settings>=2.0.3",
        "structlog>=23.2.0",
    ],
)