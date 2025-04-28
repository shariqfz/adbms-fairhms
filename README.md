## Setup

1.	git clone https://github.com/shariqfz/adbms-fairhms.git
2.	python -m venv fairhms-env
3.	Linux: source fairhms-env/bin/activate
		Windows: fairhms-env\Scripts\activate
4.	pip install -r requirements.txt

## Run
`python algo/main.py <dataset_name kept in /data dir> <dim> `

Example:
`python algo/main.py anti_2_10000_skyline.txt 2 `
