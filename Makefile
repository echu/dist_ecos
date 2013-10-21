.PHONY: clean
clean:
	find . -name "*.pyc" -delete
	find . -name "*.out" -delete
	rm parsetab.py*
