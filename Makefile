create_documentation:
	make clean
	make html

deploy:
	rm -rf dist
	rm -rf build
	python3 -m build
	python3 -m twine upload dist/*

clean:
	rm -f *.o prog3

html:
	sphinx-build -b html docs/source docs/build
