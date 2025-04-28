## Setup

1.	git clone https://github.com/shariqfz/adbms-fairhms.git
2.	python -m venv fairhms-env
3.	Linux: source fairhms-env/bin/activate
		Windows: fairhms-env\Scripts\activate
4.	pip install -r requirements.txt

Then you should end up with the following files:

- `data/`: contains two processed datasets as introduced above after attribute normalization and skyline computation. `adultSky.txt` stands for the `Adult` dataset where the dimensionality is 5, the dataset size is 32,561, and the number of the groups is 2. `anti_2_10000_skyline.txt` stands for the `Anti-Correlated` dataset where the dimensionality is 2, the dataset size is 10,000, and the number of the groups is 3.

- `result/`: is used to store the results that algorithms output.

- `utils/`: contains our utility functions sampled under different dimensions.

-	The output will be stored in the folder `result`.

## Run
`python python\main.py <dataset_name kept in /data dir> <dim> < is2D ? 1 : 0 >`

Example:
`python python\main.py anti_2_10000_skyline.txt 2 1`
