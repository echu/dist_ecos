A little more cleaned up now. Try running the examples in dist_ecos subfolder:

```python
python simple_example.py
python general_example.py
python compare_simple_general.py
```

The first script demos simple consensus, the second general consensus, and the third compares the convergence of the two methods.


Its designed to be run either serially or in parallel, to make it easier to debug.


* Todo
- move examples
- make a parallel example
- compute/use residual info (done in the last iteration, copy code to the newer code)
- make some objects/functions/things to contain the various types of solvers
- do primal/dual set intersection solversg
