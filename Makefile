.PHONY: clean
clean:
	find . -name "*.pyc" -delete
	find . -name "*.out" -delete
	find . -name "*parsetab.py" -delete
	find . -name "*graclus.edgelist*" -delete
	find . -name "mondriaan.mtx*" -delete
	find . -name "Mondriaan.defaults" -delete
