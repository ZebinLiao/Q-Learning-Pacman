# Q-Learning-Pacman
A pacman machine learning agent that utilises Q learning along with Epsilon-Greedy to play.

To run the project, head to the main folder (Q-Learning-Pacman) and execute the command:

python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid

This will train the agent for 2000 games on the small pacman grid, and do 10 recorded runs, the results will show in command line.


The agent is implemented in the QLearnAgent class in mlLearningAgents.py

To summarise:

  - the class takes in a number of inputs specifying the learning rate, elipson value for Epsilon-Greedy, discount factor for the Q learning equation, as well as the max attempts per state and number of training episodes.
  - 
  - the Q value of each state is recorded using util.Counter(), as well as the number tof time each state is visited. 
  - 
  - the Q value is continuously updated during training using the Q-learning equation.
  - 
  - the action is then chosen using a count based selection method, which means when a state has less count, it will be more likely selected for the sake of exploration. Epsilon-Greedy is also implemented to ensure exploration.

More detailed documentation is written in the code.
