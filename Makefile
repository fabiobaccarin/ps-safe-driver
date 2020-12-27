env:
	conda env create -f env.yml

split:
	python make/split.py

profiles:
	python make/profiles.py

research:
	python make/research.py

benchmark:
	python make/benchmark.py

dataset:
	python make/dataset.py

feature-list:
	python make/feature_list.py

fits:
	python make/fits.py