
import numpy as np
import cupy as cp

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

xp = np

def get_numpy(arr):
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    else:
        return arr



class Environment:
    def __init__(self, n_agents, goal_calculator, target_generator=None, dt=0.1):

        self.dt = dt

        self.goal_calculator = goal_calculator

        self.positions = xp.random.rand(2, n_agents)
        self.last_positions = xp.copy(self.positions)

        self.max_speed = 1e-2*xp.ones(n_agents)
        self.perception_radius = xp.ones(n_agents)

        #self.target_agents = np.zeros((2, n_agents), dtype=np.int)
        #self.target_agents = np.random.randint(0, n_agents-1, size=(2, n_agents))
        self.cluster_indices = xp.zeros((n_agents), dtype=np.int32)

        if target_generator is not None:
            self.target_agents, self.agent_clusters = target_generator(n_agents)
        else:
            self.target_agents = xp.random.randint(0, n_agents, size=(2, n_agents))
            self.agent_clusters = None
            
        self.goal_positions = xp.random.rand(2, n_agents)


    def update(self):
        
        # Copy last updates positions and velocities
        self.last_positions = xp.copy(self.positions)

        # Communicate with other agents
        # TODO::

        # Calculate new target position
        self.goal_positions = self.goal_calculator(self.positions, self.target_agents)
        
        diff = self.goal_positions - self.positions
        distance = xp.linalg.norm(diff, axis=0)

        vdt = self.max_speed * self.dt
        movement_mask = distance > vdt
        self.positions[:,movement_mask] += vdt[movement_mask] * diff[:,movement_mask] / distance[movement_mask]
        self.positions[:,~movement_mask] = self.goal_positions[:,~movement_mask]

    def get_average_velocity(self):
        return xp.linalg.norm(self.positions - self.last_positions, axis=0).mean() / self.dt

    #def has_converged(self):
        


def animate_positions(environment, timesteps, nframes, interval=100):
    """
    Animates the positions of agents in the environment over time.

    Parameters:
        environment (Environment): The environment object containing agent positions.
        n_frames (int): Number of frames to animate.
        interval (int): Time interval between frames in milliseconds.
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        get_numpy(environment.positions[0]), 
        get_numpy(environment.positions[1]),
        c = get_numpy(environment.agent_clusters),
        cmap='viridis',  # Color map for clusters
        )

    # Set axis limits (adjust as needed)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Agent Positions Over Time")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    avg_vel = []

    def update(frame):
        num_steps = max(timesteps // nframes, 1)
        for i in range(num_steps):
            environment.update()  # Update the environment
            avg_vel.append(environment.get_average_velocity())  # Get average velocity
        scatter.set_offsets(get_numpy(environment.positions.T))  # Update scatter plot data
        print(f"Frame {frame} \r", end="")  # Print frame number
        return scatter,


    anim = FuncAnimation(fig, update, frames=nframes, interval=interval, blit=True, repeat=False)
    plt.show()

    avg_vel = xp.array(avg_vel)
    
    plt.figure()
    plt.plot(get_numpy(avg_vel))
    plt.title("Average Velocity Over Time")
    plt.show()


def between_goal_calculator(positions, target_agents):

    p1 = positions[:, target_agents[0]]
    p2 = positions[:, target_agents[1]]

    goal_method = "less-stupid-behind"
    if goal_method == "midpoint":
        pgoal = 0.5 *(p1 + p2)
        return pgoal
    elif goal_method == "stupid-behind":
        pgoal = p1
        return pgoal
    elif goal_method == "less-stupid-behind":
        pgoal = p1 + 0.05*(p1 - p2)
        return pgoal


def target_generator(n_agents, max_n_clusters=10):
    """
    Generates random target agents for each agent in the environment.

    Parameters:
        n_agents (int): Number of agents.
        n_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Random target agents for each agent.
    """
    cluster_indices = xp.random.randint(0, max_n_clusters, size=n_agents)
    target_agents = xp.zeros((2, n_agents), dtype=np.int32)
    agent_clusters = xp.zeros((n_agents,), dtype=np.int32)
    
    for i in range(max_n_clusters):
        indices = xp.where(cluster_indices == i)[0]
        if indices.shape[0] > 1:
            choice_one = xp.random.choice(indices, size=len(indices), replace=False)
            # We are not allowed to choose ourself
            choice_mask = choice_one == indices
            choice_one[choice_mask] = indices[~choice_mask][:xp.sum(choice_mask)]   

            choice_two = xp.random.choice(indices, size=len(indices), replace=False)
            # We are not allowed to choose ourself or the first choice
            choice_mask = xp.logical_or(choice_two == indices, choice_two == choice_one)
            choice_two[choice_mask] = indices[~choice_mask][:xp.sum(choice_mask)]

            target_agents[0, indices] = choice_one
            target_agents[1, indices] = choice_two

            agent_clusters[indices] = i
    
    return target_agents, agent_clusters


if __name__ == "__main__":
    n_agents = 5000
    timesteps = 1000
    nframes = 8000

    #target_agents = target_generator(n_agents, 10)

    env = Environment(n_agents, between_goal_calculator, lambda n: target_generator(n, 10))

    animate_positions(env, timesteps, nframes, interval=1)