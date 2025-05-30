---
author:
- Jakob Galley, Viktor Vigren Näslund
date: April 2025
title: WASP AIML
---


# Its probaly easiest to read the report on the github page. Link to the project git is [here](https://github.com/turbotage/AI-ML-Course/)

# Introduction

In Örebro, we did a human swarm experiment.
Participants had to choose targets based on some distrubition and had to find a way to fulfill their task in the room.
In that problem setting there are underlying biases as target distribution, speed and perception of the participants.
But nevertheless, after some time, we all arrived at some kind end state.
This state was mostly static, but in one task (cf. task(b)), we a got a dynamic periodic end state.

The aim of this exercise was to study and maybe repdroduce the insights gathered in the human swarm experiment.
By modelling this problem with a computer, we can change parameters as number of agents or velocity more freely.
It will fail to modell humans exactly but gives a way to do multiple experiments in a short time with  parameters which we cannot fathom to do in the real world.

# Problem setting

Develop a simulation of the human swarm exercise in a programming
language of your choice. Can you reproduce the overall behavior patterns
that we observed during the experiment? Analyse convergence and
sensitivity of the swarm behavior to different parameter settings, such
as perception radius, heterogeneity of speed, etc. . Assume that the map
is free of obstacles. The simulation should reproduce the following
positioning scenarios:

1.  Every agent $X$ selects two other agents randomly: $A$ and $B$ (and
    memorizes them). During the experiments agent $X$ always tries to
    position itself between $A$ and $B$. Every agent does the same.
    Decision making of the agents must be strictly local.

2.  Initial setup is the same as in a. The difference is that $X$ now
    tries to position itself so that $B$ is between $A$ and $X$ ( $X$
    hides behind $B$, so that $A$ cannot attack/see\...).

3.  Would anything change if agents are allowed to communicate? Can you
    imagine a scenario in which communication would be important?

Prepare a report documenting your model(s) as well as giving details of
your simulator.

# Code presentation

We decided to code our simulator in Python because of familiarity, ease
of use and flexibility. As the logic of our agents is not to complicated
we calculated their decision and updated their position globally. Thus,
we only have an environment class which holds the necessary functions
for initialization and updating. Furthermore, we have outside help
methods for target generation, goal calculation and plotting.

## Environment class

On initialization, we have to give this class the number of agents, the
goal calculator function and a target generator. Furthermore, we can
change the timestep size dt and the perception radius. We store these
parameters and initialize random positions for our agents which we store
as an array. Also, the agents speed is initialized as the shift of the
absolute value of an Gaussian array.

In the update method, the agents position gets updated based upon the
goal function, perception radius and timestep size. If the target's
position is outside of the perception radius, then the last memorized
position is used for goal calculation. For the goal calculation, we use
the passed goal calculation function. This function should return the
goal positions for our agents as an array. Then, our agents move towards
these positions as far as possible depending on their speed and the
timestep size.

Here is the pseudo-code: Calculate the distance between the goal
positions and the agents positions 

```python
diff = pgoal - positions
dist = np.norm(diff)
```

Then, we update the positions based upon the distance is
reachable within the time step in the following way:

```python
new_positions = pgoal if vdt >= diff else vdt/dist
```

Then, there is also a small function which returns the average velocity
between each time step using the last and new position of the agents.

## Goal calculation

We use a ```goal_calculator``` function which gets passed to the
position of the agents ```positions```, the targets positions ```p1, p2``` and a string ```goal_method``` to decide the
goal method. As an output, we return the goal positions of all the
agents as an array.

We have the following options for goal calculations:

**\"midpoint\"**:  the goal is to position oneself in the middle between
    both targets.

```python
pgoal = (p1 + p2)/2
```

**\"inbetween\"**:  the goal is to position oneself on the closest point
    in the line segment between the targets. For that, we calculate
    scalar projection of our vector onto the line and then by checking
    mininima and maxima with regards to $0$ and $1$, calculate the
    nearest point in the line segment. The pseudo code looks like this:
    Calculate the scalar projection:


```python
dir = p1 - p2
norm = np.norm(dir)
s = np.dot(positions - p2, dir)/norm
```
    
Then the nearest point on the line segment is given
by 

```python
t = np.max(np.min(s,1), 0)
pgoal = p2 + t*dir
```


If the norm is $0$, we get a division by zero. In
this case both targets are on the same point, so just try to get to
this point.

**\"tailgating\"**: the goal is to position to oneself to the closest
point in the line behind target2. This is similar to \"midpoint\"
with the difference being in the possible line parameter we are
looking for ($(-\infty, 0]$ instead of $[0, 1]$). 

```python
t = np.min(s, 0)
pgoal = p2 + t*dir
```


## Target generation

First, we thought that every possible target system can be explained
through cycles and agents who are not part of these cycles but have one
or more targets in a cycle. That is why first generated a method to
generate cycles and a method for singletons who target agents in these
clusters. Then we noticed that this assumption is flawed and just
created a random target generator method. Here we used some clever
modulo arithmetic to guarantee that every agent has two different
targets and does not target themself.

We set up the identity array as

```python
I = np.arrange(0, n_agents) # I = [0, 1, .. , n_agents-1]
```

Then, if we add a random integer array with values between $1$ and ```n_agents``` 
with ```numpy.randInt``` and take the result modulo ```n_agents```, we cannot 
get the identity:

```python
shift1 = np.randInt(1, n_agents)
target1_index = (I + shift1)%%n_agents
```

For the second target, we need to guarantee a new shift
between $1$ and ```n_agents``` which is different to ```shift1```
at every position. We used the same trick as above, but with modulo
```n_agents-1``` to guarantee a new shift. Because we need to $1$ at
the end, the random integer array is allowed this time to take values
between $0$ and ```n_agents-2```. Note here, that an addition of
```n_agents-1``` corresponds to subtracting $1$ in modulo arithmetic
which becomes $0$ when adding $1$ at the end. Thus 

```python
shift2 = (shift1 + np.randInt(0, n_agents-2))%%n_agents
shift2 = shift2 + 1
target2_index = (I + shift2)%%n_agents
```

## Animation/Plotting
Each run, produces an gif/animation of the simulation, a relational graph over agents and their chosen targets and a plot of the average velocity over time.
For the average velocity plot and animation, we used ```matplotlib```, and for the plot of the relational graph, we used ```networkx```.

In the animation, each agent is colored. The meaning of the coloring is dependent on which target generation method was used. For the ```random_generator``` the coloring has no meaning, while for the ```cluster_generator``` each cluster gets its own color.
None of the examples in this text will be using the ```cluster_generator``` but feel free to test it. Each agents also has an arrow attached. The arrow points towards its current goal position. 

## Iterative Procedure
How each agent shall move towards its goal position is determined by the ```update()``` function of the ```Environment``` class.

The ```Environment``` class holds the ```memory_positions``` array and the ```memory_length``` array. ```memory_positions[:,i,j]``` holds the last position agent ```i``` saw agent ```j```. That is the last position of agent ```j``` when it was still in the perception radius of agent ```i```. 
```memory_length[i,j]``` holds how many iterations ago, it was that agent ```i``` saw agent ```j```.

At the beginning of every new ```update()``` the memory length is incremented. Distances between all agents are calculated.
Then we calculate which are within ```communication_radius``` and which are in ```perception_radius```.
For all agents that are within ```perception_radius``` of eachother, ```memory_positions``` and ```memory_length``` is updated.

We then iterate over all agents. For each agent we extract all other agents that are within that agents communication radius (This includes the agent itself). We then checks which agent within the communication radius, that has the freshest memory of this agents targets. 
I.e the freshest memory position of the target agents, becomes the positions we later calculate our goal position with.
These positions are the ```p1``` and ```p2``` arrays in the code. Note that we can eliminate communication by setting ```communication_radius``` to zero. Then the only agent within communication radius is the agent itself. And it will update ```p1``` and ```p2``` for goal setting, only based on its own perception radius.

When ```goal_positions``` has been calculated from ```p1``` and ```p2```, we move towards it. 
This is done by taking a step of length ```speed * dt``` in the direction of the goal. 
If ```speed * dt``` is longer than the distance to the goal position, we move straight to it.

The ```memory_positions``` are initialized to a random point inside a boxed with sizelength $0.5$ centered in the room. This means that in the case of small perception radius. All agents will begin to travel towards the center and hopefully find their targets.


# Experiments

## Task (a)

In task (a), we considered the \"in-between\" and \"mid-point\" goal
calculation method, which tries to place the agents either on the nearest
point of the in-between-line of the target agents or in the middle. In
both cases, results seemed to coincide differing only in the speed of
convergence.

We ran multiple tests with varying number of agents, but keeping everything else constant.
Importantly, the perception radius was chosen as big enough, such that each agent can see all other agents.

Indeed, in all our experiments the system converged to stable state.
These states are characterized by non-moving agents usually huddling
together, forming clusters and converging to single points. 
Inbetween these clusters, lines can form (cf. 200 agents in the section on perception radius).
Here, the number of agents influences how many clusters and lines will form more regularly.
In some cases the end results was a little bit dynamic, but this is can be
attributed to step size of the agents.

This seems to coincide with the results of our human swarm experiments, where large 
groups of people huddled together trying to push eachother away and lines were forming inbetween these clusters.
The difference to our simulation is, that agents can go through eachother and cannot be prevented from moving to their target location.

The following example is of 25 agents which use the midpoint calculation:

![Midpoint full perception animation](saved_gifs/midnew/2025-05-19_midpoint_nagents_25_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_900_nf_450_gif.gif) 

![Midpoint full perception velocity](saved_gifs/midnew/2025-05-19_midpoint_nagents_25_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_900_nf_450_plt.jpg)

![Midpoint full perception graph representation](saved_gifs/midnew/2025-05-19_midpoint_nagents_25_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_900_nf_450_graph.jpg)

Note that the average velocity is not necesarrily monotone falling. This is due to small changes of targets increasing the agents goalposition drastically.
This can be seen even better with the "inbetween" method, which is not as linear dependent on the target agents.
The following example is of 25 agents which use the inbetween calculation:

![Midpoint full perception animation](saved_gifs/midnew/2025-05-19_inbetween_nagents_25_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_900_nf_450_gif.gif)

![Midpoint full perception velocity](saved_gifs/midnew/2025-05-19_inbetween_nagents_25_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_900_nf_450_plt.jpg)

![Midpoint full perception graph representation](saved_gifs/midnew/2025-05-19_inbetween_nagents_25_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_900_nf_450_graph.jpg)

In both examples, some clusters seem to form quickly with other agents joining after.
We believe that these first clusters collapses are due what we call minimial dynamical systems.
These are minimal subsets of agents which are closed under the choice of targets.
For more information see our appendix, where tried to mathematically formulate this problem.

## Task (b)

In task (b), we considered the \"tailgating\" goalcalculation method, which tries to place the agent to the closest point on the line behind target agent 2.
Suprisingly, this resulted into a similar behaviour as in task (a) with some occurences of dynamical periodic end states.

Here, we also kept everything but the number agents constant and choose the perception radius big enough such that every agent see everyone else.

The convergence process can be very different to task (a).
We see a lot of lines forming after slower agents which then follow these slower agents until either everything collapses to points or they form a cyclic periodic shape.
The latter one can be somewhat attributed to the stepsize.

In the human swarm experiment, we saw a periodical endstate forming.
So, we expected a lot more of these big periodical end states but maybe due to our goal calculation method, these didn't arise as often as anticipated.
But looking back, the endstate in our human swarm experiment would have collapsed into endstates which we saw during testing: a lot of agents exchanging their position rapidilly around a point.

The following example is of 25 agents which use the tailgating calculation:

![Tailgating full perception animation](saved_gifs/midnew/2025-05-19_tailgating_nagents_25_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_900_nf_450_gif.gif)

![Tailgating full perception velocity](saved_gifs/midnew/2025-05-19_tailgating_nagents_25_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_900_nf_450_plt.jpg)

![Tailgating full perception graph representation](saved_gifs/midnew/2025-05-19_tailgating_nagents_25_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_900_nf_450_graph.jpg)

Note how the end velocity is not $0$ - the agents are fluctuating near a point.

Here is a more clear example of a periodic shape forming.
At the end, it will probably converge to a point, but in a more realistic setting, this would be a periodic pattern (200 agents)

![Tailgating periodic gif](saved_gifs/special%20renders/2025-05-19_tailgating_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_0_gif.gif)

![Tailgating periodic plt](saved_gifs/special%20renders/2025-05-19_tailgating_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_0_plt.jpg)

![Tailgating periodic graph](saved_gifs/special%20renders/2025-05-19_tailgating_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_0_graph.jpg)

## Task (c)

For task (a) and (b) the communication radius was set to zero, effectivly turning off communication. In this task, we vary the communication radius to try and understand the effect of communication. To show the potential drastic differences, we first keep the communication radius small and then increment it.

Below is an example of a super small perception radius, and zero communication. As can be seen,
when the perception radius is this small and we have no communication, the agent will likely never see its target and just continue to the initialized starting memory position.

![inbetween](saved_gifs/comms/2025-05-21_inbetween_nagents_200_mu_pr_0.001_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_6000_nf_350_gif.gif)

This straight motion to initialized starting memory position can also be understood from the velocity below.
Iff all particles move straight to the starting memory position. The velocity would just taper off like below.

![inbetween](saved_gifs/comms/2025-05-21_inbetween_nagents_200_mu_pr_0.001_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_6000_nf_350_plt.jpg)

We then turn on the communication, with a small communication radius of $0.05$. This creates a more tightly clustered end result. Indicating that more agents find their true goal position. What can also be seen is a much more chaotic end behaviour. Some agents seem to walk around or "bounce" around near the end state. This is a communication type behaviour. What happens is that the agent walks towards its current goal position. It then ends up within communication radius of another agent with a more recent memory of one of the agents targets. This renews the goal position of the agent so it changes direction.

![inbetween](saved_gifs/comms/2025-05-20_inbetween_nagents_200_mu_pr_0.001_std_pr_0.0_mu_cr_0.05_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_6000_nf_350_gif.gif)

![inbetween](saved_gifs/comms/2025-05-20_inbetween_nagents_200_mu_pr_0.001_std_pr_0.0_mu_cr_0.05_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_6000_nf_350_plt.jpg)

One thing that we didn't add, but which would make the communication more powerful. Is the update of the agents own last memory position based on the last memory position of other agents within communication radius.
So that if another agent within communication radius had a more recent memory position. Then the agents memory position would be updated to that. Of course, one could also make the communication into a network. So that asking one agent, that agent could then ask other inside their communication radius and so on. In that case given enough agents. The effective communication radius would be much larger.

# Influence of parameters

## Percpetion radius

The perception radius can influence the simulation quite a lot.
The agents tend to move towards the middle because of initialized memory.
But then, they only become active as soon as their targets come into field of vision.
When this happens it can trigger a chain reaction with other agents becoming active as a result of the agent moving.
Overall, the end state becomes more cluttered.

Have for example a look at these simulations (full perception, 0.5, 0.2, 0.0001) with 200 agents for the inbetween method

![Inbetween full perception](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Inbetween half perception](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_0.5_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Inbetween fifth perception](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_0.2_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Inbetween fifth perception](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_0.0001_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

and here is how the spread influences the simulation (mean=0.2, std=0.1/0.2/0.3/1.0)

![Inbetween with more spread](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_0.2_std_pr_0.1_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Inbetween with even more spread](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_0.2_std_pr_0.2_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Inbetween with even even more spread](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_0.2_std_pr_0.3_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Inbetween with even even more spread](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_0.2_std_pr_1.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

We get a lot more active agents, but some less perceptive agents still get lost.

And for the tailgating method, we see similiar results:

![Tailgating full perception](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Tailgating half perception](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_0.5_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Tailgating fifth perception](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_0.2_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

And this is how the spread influences the simulation (std=0.1, 0.2, 0.3, 1.0 with mean=0.2)

![Tailgating with more spread](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_0.2_std_pr_0.1_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Tailgating with even more spread](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_0.2_std_pr_0.2_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Tailgating with even even more spread](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_0.2_std_pr_0.3_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Tailgating with even even more spread](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_0.2_std_pr_1.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

## Velocity

Changing the mean just changes the speed of convergence, so it is more interesting to look how the spread of the velocity impacts the simulation.
Here is an example, where the velocity drops from 0.1 to 0.01:

![Standard inbetween](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Standard inbetween](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.01_std_speed_0.0_ndt_1200_nf_600_gif.gif)

When we change the spread of the overall velocity, then there exist now some agents who are faster than others.
These agents we move more quickly to their target location and might have to wait for their slower targets to move.

![Standard inbetween](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.01_std_speed_0.05_ndt_1200_nf_600_gif.gif)

![Standard inbetween](saved_gifs/midnew/2025-05-19_inbetween_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.01_std_speed_0.1_ndt_1200_nf_600_gif.gif)

And we can see the same in the case of task (b)

![Standard tailgating](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_1200_nf_600_gif.gif)

![Standard tailgating](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.01_std_speed_0.0_ndt_1200_nf_600_gif.gif)

and spread

![Standard tailgating](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.01_std_speed_0.05_ndt_1200_nf_600_gif.gif)

![Standard tailgating](saved_gifs/midnew/2025-05-19_tailgating_nagents_200_mu_pr_5.0_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.01_std_speed_0.1_ndt_1200_nf_600_gif.gif)

## Communication Radius

The effect of communication radius is large. Its hard to make a fair comparison, but for ending up at your goal position, it is often better to have a quite small perception radius but large communication radius than having a large perception radius. To examplify this we below first show an animation of agents having a perception radius of $0.1$ then an animation of agents having $0.001$ as perception radius but $0.1$ communication radius.

As can see in the $0.1$ perception radius case below. Alot of the agents don't converge to their true goal position.
![Standard tailgating](saved_gifs/comms/2025-05-20_inbetween_nagents_200_mu_pr_0.1_std_pr_0.0_mu_cr_0.0_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_6000_nf_350_gif.gif)

While in the example below of $0.001$ perception radius but $0.1$ communication radius, a much larger portion of the agents converge.

![Standard tailgating](saved_gifs/comms/2025-05-21_inbetween_nagents_200_mu_pr_0.001_std_pr_0.0_mu_cr_0.1_std_cr_0.0_mu_speed_0.1_std_speed_0.0_ndt_6000_nf_350_gif.gif)


# Appendix: Mathematical formulation

## Task (a)

The start setting of our problems can be modelled as simple direct
graphs where the set of vertices $V$ is the set of agents and two
vertices $v_1$ and $v_2$ are connected by edge $e$ in direction $v_1v_2$
iff $v_1$ has $v_2$ as its target agent. We will denote by $E$ the set
of all directed edges.

For a vertex $v$, we will denote by $\deg^+(v)$ the *outdegree* and by
$\deg^-(v)$ the *indegree* of the vertex $v$. In our case $\deg^+ = 2$,
and thus

$$2\cdot|V| = \sum_{v\in V} \deg^+(v) = \sum_{v\in V} \deg^-(v) = |E|.$$

We call $V'\subseteq V$ a *dynamical system* if the induced subgraph can
be modelled as a result of the same problem setting, i.e. the induced
subgraph is only depended on itself. This is equivalent to the following
easy criteria

$$2\cdot |V'| = |E(V')|.$$

We can the sets of dynamical
systems with the set order $V_1 \leq V_2$ iff $V_1 \subseteq V_2$. We
call $V'\subseteq V$ a *minimal dynamical system* if it is minimal with
regards to this order.

Let us consider the free $\mathbb{R}$-vector space $F(V)$ of $V$, which
has as elements finite formal sums of the following form

$$y = \sum_{v\in V} \lambda_v v$$ 

with $\lambda_v\in \mathbb{R}$. There is a canonical embedding $\phi$ of $V$ into $F(V)$ given by
$\phi(v) = 1v$. Furthermore, if, as in our case, $|V|<\infty$, then
$F(V) \simeq \mathbb{R}^{|V|}$

For a directed graph $(V, E)$ we can also define an adjacency matrix $A$
with 

$$
A_{vw} = \begin{cases}
        1 & wv \in E \\
        0 & \text{else}.
    \end{cases}
$$
    
Note that for a dynamical subsystem of
$V'\subseteq V$, the induced adjacency matrix is just the restriction of
$A$ to $V'$. Furthermore, if $V$ is a minimal dynamical system, then the
row and column sum of $A$ is $2$. Note that $A$ can be viewed as a map
from $V$ into $F(V)$ which maps $w$ onto the sum of $v$'s with
$vw\in E$. This map extends then uniquely to a linear map from $F(V)$
onto itself given by 

$$Ay = \sum_{v, w\in W} A_{vw} \lambda_w v$$

Each vertex $v$ in $V$ has a position feature $x_v\in \mathbb{R}^2$.
This map extends uniquely as a linear map on $F(V)$ given by

$$x_{\sum_{v\in V} \lambda_v v} = \sum_{v\in V} \lambda_v x_v.$$

So $x$ can be viewed as a matrix of the form $\mathbb{R}^{2\times |v|}$. As a
result, $x$ behaves nicely with linear transformations of $F(V)$

$$x_{Ty} = \sum_{v, w\in V} T_{vw} \lambda_w x_v = Tx_y$$ 

for any $y\in F(V)$.

We write $x_V = (x_v)_{v\in V}$ as the vector containing the canonical
basis of $F(v)$.

In task $(a)$ we would like to minimize the following goal function

$$\mathcal{E}(x_V) = \sum_{v\in V} \left\|x_v - \frac{1}{2}x_{Av} \right\|^2 = \sum_{v\in V} \left\|\left(I - \frac{1}{2}A\right)x_v \right\|^2$$

with the start position of all the vertices of $V$ given by a matrix
$X_0$ 

$$x_V(0) = X_0.$$ 

This minimization problem corresponds to the gradient flow equation 

$$\partial_t x_V(t) = - \nabla\mathcal{E}(x_V).$$

The derivative of $\mathcal{E}$ is given by
$\nabla\mathcal{E}(x_V) = \left(2I - A - A^T + \frac{1}{2}A^TA\right) x_V$.
By setting

$$T = 2I - A - A^T + \frac{1}{2}A^TA = 2\left(I-\frac{1}{2}A^T\right)\left(I-\frac{1}{2}A\right),$$

the uniquely solution of the Gradient Flow is given by

$$x_V(t) = X_0 \exp(-tT).$$ 

Note that for a minimal dynamical system,
the matrices $I-\frac{1}{2}A$ and $I-\frac{1}{2}A^T$ are positive
semi-definite as the diagonal is $1$ and the sum of rows and columns is
always $0$. In particular, $T$ has only eigenvalues which are greater or
equal to $0$. As a result, $\exp(-tT)$ behaves as a constant matrix as
$t\to \infty$ and there exists a unique constant end state
$x_V(\infty)$.

This end state must be a collapse of all points to one point. In this
way all minimal dynamical systems are classified.

In our similation, we don't use gradient descent. Although the for a minimal dynamical system. The constant end state should still hold with our method. For the inbetween and the midpoint method the iterative procedure comes from a contraction mapping (a nonexpansive and one strictly decreasing). But the gradient descent formulation also illuminates a multiagent system where another non local behaviour of the agents and communication would move towards a solution. That is if all the agents could communicate their positions. The direction an agent should move could be given by the gradient descent direction for the whole population minima objective instead of a local movement towards their own goal position.
