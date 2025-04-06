import nox

nox.options.sessions = ["lint", "tests"]


@nox.session(python=["3.9", "3.10", "3.11"])
def tests(session):
    """Run the test suite."""
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.run("pytest", *session.posargs)


@nox.session
def lint(session):
    """Lint using flake8."""
    session.install("flake8", "flake8-docstrings")
    session.run("flake8", "nova", "tests")


@nox.session
def black(session):
    """Run black code formatter."""
    session.install("black")
    session.run("black", "nova", "tests")


@nox.session
def isort(session):
    """Run isort import sorter."""
    session.install("isort")
    session.run("isort", "nova", "tests")


@nox.session
def mypy(session):
    """Run mypy type checker."""
    session.install("mypy", "types-all")
    session.install(".")
    session.run("mypy", "nova", "tests")


@nox.session
def docs(session):
    """Build the documentation."""
    session.install("sphinx", "sphinx-autodoc-typehints")
    session.install(".")
    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")
