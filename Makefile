create_documentation:
	cd docs && make clean
	sphinx-apidoc -f -o docs/source/ spark_matcher/
	cd docs && make html

deploy:
	rm -rf dist
	rm -rf build
	python3 -m build
	python3 -m twine upload dist/*