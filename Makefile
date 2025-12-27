.PHONY: clean venv test package build install

clean:
	@rm -rf ./dist/* ./build/* ./*.egg-info

venv:
	@python3 -m venv venv && source ./venv/bin/activate && pip3 install -r requirements.txt

test:
	@pip3 install -q -r requirements.txt &&\
	  cd test &&\
	  python3 -m pytest -s

# Build package using modern pyproject.toml (recommended)
build:
	@echo "Building package with uv..."
	@uv build
	@echo "Package built successfully in dist/"

# Legacy build using setup.py (for backwards compatibility)
package:
	@echo "Building package with setup.py (legacy)..."
	@python setup.py sdist
	@echo "Package built successfully in dist/"

# Install the package locally in editable mode
install:
	@echo "Installing package in editable mode..."
	@uv pip install -e .