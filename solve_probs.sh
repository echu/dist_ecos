#!/usr/bin/env sh
TODAY=`date "+%Y-%m-%d"`
./test_fixed_part.py --infile data-25x5x256.p --outfile $TODAY-25x5x256.p --noplot
./test_fixed_part.py --infile data-50x10x256.p --outfile $TODAY-50x10x256.p --noplot
./test_fixed_part.py --infile data-250x50x256.p --outfile $TODAY-250x50x256.p --noplot
./test_fixed_part.py --infile data-500x100x256.p --outfile $TODAY-500x100x256.p --noplot
./test_fixed_part.py --infile data-2500x500x256.p --outfile $TODAY-2500x500x256.p --noplot
