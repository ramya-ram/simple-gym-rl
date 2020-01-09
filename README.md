## Simple tabular RL with OpenAI gym

### Installation:

* Clone this repository

	```
	git clone https://github.com/ramya-ram/simple-gym-rl.git
	cd simple-gym-rl
	```

* Set up an Anaconda environment with the required packages:
	```
	conda env create -f environment.yml
	source activate simple-gym-rl
	```

* Install the domains in this repo:

	```
	cd domains
	pip install -e .
	```

### Running the code:

* To train the agent in a particular environment, run:

	```
	cd run_q_learning
	python run_game.py (game-name) (save-directory-name)

	e.g.
	python run_game.py "MyCatcher-v0" mycatcher
	```

	The learned Q-values, mean reward learning curve, and other debug info (state counts, etc.) will be saved to the specified directory location.
	The domain code for MyCatcher is located in domains/domains/ple. The code for the Q-learning part is in the run_q_learning folder and includes run_game.py and q_learner.py.

* If you want to watch the agent play the learned source/target tasks, run:

	```
	cd run_q_learning
	python run_game.py (game-name) (save-directory-name) (learned-Q-file)

	e.g.
	python run_game.py "MyCatcher-v0" mycatcher_learned mycatcher/Q.csv
	```