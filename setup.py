from setuptools import find_packages, setup

setup(
    name="nova",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "playwright>=1.40.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    python_requires=">=3.9",
)
