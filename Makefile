create_documentation:
	cd docs
	make clean && make html

deploy:
	rm -rf dist
	rm -rf build
	python3 -m build
	python3 -m twine upload dist/*