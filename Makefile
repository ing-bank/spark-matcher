create_documentation:
	cd docs && make clean
	sphinx-apidoc -f -o docs/source/ spark_matcher/
	cd docs && make html
