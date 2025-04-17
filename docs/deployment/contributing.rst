Contributing
============

Thank you for your interest in contributing to Nova! This document provides guidelines and instructions for contributing to the project.

Code of Conduct
--------------

By participating in this project, you agree to abide by our `Code of Conduct <CODE_OF_CONDUCT.md>`_.

Getting Started
--------------

1. Fork the repository
2. Clone your fork::

      git clone https://github.com/your-username/nova.git

3. Create a virtual environment::

      python -m venv venv

4. Activate the virtual environment:

   - Windows::

        venv\Scripts\activate

   - Unix/MacOS::

        source venv/bin/activate

5. Install development dependencies::

      pip install -e ".[dev]"

6. Install Playwright browsers::

      playwright install chromium

Development Workflow
------------------

1. Create a new branch for your feature/fix::

      git checkout -b feature/your-feature-name

2. Make your changes
3. Run tests::

      pytest

4. Run linting and type checking::

      make lint
      make typecheck

5. Commit your changes::

      git commit -m "Description of changes"

6. Push to your fork::

      git push origin feature/your-feature-name

7. Create a Pull Request

Code Style
---------

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings following Google style
- Keep functions small and focused (max 50 lines)
- Use meaningful variable and function names
- Add comments for complex logic
- Follow the project's import order convention

Testing
-------

- Write unit tests for all new features
- Use pytest fixtures for test setup
- Mock external dependencies (LLM, browser, etc.)
- Ensure test coverage remains above 90%
- Use descriptive test names
- Group related tests in classes
- Add integration tests for critical paths

Documentation
------------

- Update relevant documentation files
- Add docstrings to all new functions and classes
- Update examples if API changes
- Keep architecture diagrams up to date
- Document configuration options
- Add troubleshooting guides for new features

Pull Request Process
------------------

1. Ensure your PR description:
   - Clearly describes the problem and solution
   - Lists all changes made
   - References related issues
   - Includes test results

2. Include:
   - Unit tests for new features
   - Integration tests for critical paths
   - Updated documentation
   - Type hints and docstrings

3. Ensure:
   - All tests pass
   - Code is properly formatted
   - Type checking passes
   - Documentation is updated
   - No linting errors

4. Request review from maintainers

Project Structure
---------------

- ``src/nova/``: Main package code
  - ``agents/``: Agent implementations
  - ``core/``: Core components
  - ``tools/``: Tool implementations
  - ``config.py``: Configuration management

- ``tests/``: Test suite
  - ``unit/``: Unit tests
  - ``integration/``: Integration tests
  - ``fixtures/``: Test fixtures

- ``docs/``: Documentation
  - ``api/``: API reference
  - ``user-guides/``: Usage guides
  - ``architecture/``: Architecture docs
  - ``deployment/``: Deployment guides

Questions?
----------

Feel free to open an issue if you have any questions or need clarification. 