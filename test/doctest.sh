#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Running all tests in $DIR/../src/"
python -m doctest -v $DIR/../src/**/*.py