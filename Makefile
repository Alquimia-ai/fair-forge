.PHONY: clean venv test package

clean:
	@rm -rf ./dist/*

venv:
	@python3 -m venv venv && source ./venv/bin/activate && pip3 install -r requirements.txt

test:
	@pip3 install -q -r requirements.txt &&\
	  cd test &&\
	  python3 -m pytest -s

package:
	@python setup.py sdist