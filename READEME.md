

## Project Description
> The objective of this project is to use model-free RL techniques to implement the **Frozen Lake Problem** and its extensions. 
> The problem is essentially a grid-world situation in which the agent’s target is to go from the start point(initial state) and reach the frisbee(goal & terminal state), 
> while avoiding slipping into ice holes(terminal states) on the frozen lake.

Python is typically used to build three algorithms using various variable structures:**First-Visit Monte Carlo**,**SARSA**, and **Q-learning**, where these methods are designed to find an optimal policy(i.e., _for control_)

## Detailed Interpretation in Project Execution
### Map Generation & Environment Construction
The folder `map` contains what basic files need to construct environment. 
The `.txt` file  in `\map` contains the 4✖4 map and 10✖10 map: 
0 represents ice surface that agent be able to go though, 1 represents ice holes, 2 represents start point, 3 represents frisbee.
The visualization map can be illustrated as :

<img alt="frozenlake" height="10" src="map/map_4x4.png" width="10"/>
<img src="map/map_4x4.png" width="40%" alt="map_4x4"/>
<img src="D:\ME5406_FrozenLake\map\map_4x4.png" width="40%" alt="map 4x4"/>

