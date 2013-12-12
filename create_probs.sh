#!/usr/bin/env sh
./random_cone_problem_fixed_part.py -m 25 -n 5 -N 256 -p 0.4
./random_cone_problem_fixed_part.py -m 50 -n 10 -N 256 -p 0.2
./random_cone_problem_fixed_part.py -m 250 -n 50 -N 256 -p 0.2
./random_cone_problem_fixed_part.py -m 500 -n 100 -N 256 -p 0.1
./random_cone_problem_fixed_part.py -m 2500 -n 500 -N 256 -p 0.1