#!/bin/bash

dts devel build -f
dts devel run -R csc22941 -L motion
python3 packages/my_package/src/shutdown.py