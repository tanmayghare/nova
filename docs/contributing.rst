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

      pip install -r requirements-dev.txt

6. Install Playwright browsers::

      playwright install

Development Workflow
------------------

1. Create a new branch for your feature/fix::

      git checkout -b feature/your-feature-name

2. Make your changes
3. Run tests::

      pytest

4. Format code::

      black .
      isort .

5. Commit your changes::

      git commit -m "Description of changes"

6. Push to your fork::

      git push origin feature/your-feature-name

7. Create a Pull Request

Code Style
---------

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Write tests for new features

Testing
-------

- Write unit tests for new features
- Ensure all tests pass before submitting a PR
- Use descriptive test names
- Mock external dependencies in tests

Documentation
------------

- Update README.md if necessary
- Add docstrings to new functions and classes
- Update examples if API changes

Pull Request Process
------------------

1. Ensure your PR description clearly describes the problem and solution
2. Include relevant tests
3. Update documentation if necessary
4. Ensure all CI checks pass
5. Request review from maintainers

Questions?
----------

Feel free to open an issue if you have any questions or need clarification. 