PATH_SP = ~/miniconda3/envs/ds/lib/python3.8/site-packages/
PATH_PD = pandas/core/dtypes/missing.py

all:
	@echo "Edit path to python site-packages in Makefile and run 'make patch'"

requirements:
	pip install -r requirements.txt

patch: requirements
	patch $(PATH_SP)$(PATH_PD) patch.diff
