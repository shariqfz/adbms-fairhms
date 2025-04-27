
### Usage

First clone this repository:

	git clone https://github.com/dcsjzh/FairHMS.git

Then you should end up with the following files:

- `data/`: contains two processed datasets as introduced above after attribute normalization and skyline computation. `adultSky.txt` stands for the `Adult` dataset where the dimensionality is 5, the dataset size is 32,561, and the number of the groups is 2. `anti_2_10000_skyline.txt` stands for the `Anti-Correlated` dataset where the dimensionality is 2, the dataset size is 10,000, and the number of the groups is 3.

- `executable/`: contains the executable file.

- `result/`: is used to store the results that algorithms output.

- `utils/`: contains our utility functions sampled under different dimensions.

- `src/`: contains the `C++` implementation of all algorithms.

Assuming that you followed the instructions above, in order to run the algorithms, you need to execute the following steps:

a. Compilation

	cd src
	make

b. Execution

	cd ../executable

Follow the command format below, you could run all algorithms with a specific number of categories, a specific k on the dataset and a 2D dataset flag is2D.

	./run.out dataset k is2D 

e.g., for high-dimensional dataset:

	./run.out adultSky.txt 10 0

e.g., for two-dimensional dataset:

	./run.out anti_2_10000_skyline.txt 5 1
	
c. Output

	The output will be stored in the folder `result`.
