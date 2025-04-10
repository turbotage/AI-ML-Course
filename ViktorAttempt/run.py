
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Environment:
    def __init__(self, n_agents, goal_calculator):

        self.goal_calculator = goal_calculator

        self.positions = np.random.rand(2, n_agents)
        self.max_speed = 1e-2*np.ones(n_agents)
        self.perception_radius = np.ones(n_agents)

        self.target_agents = np.random.randint(0, n_agents, size=(2, n_agents))
        #self.target_agents = np.ascontiguousarray(np.array([[1,2],[0,2],[0,1], [4,5],[3,5],[3,4]]).transpose())

        self.goal_positions = np.random.rand(2, n_agents)


    def update(self):
        self.goal_positions = self.goal_calculator(self.positions, self.target_agents)

        diff = self.goal_positions - self.positions
        distance = np.linalg.norm(diff, axis=0)
        speed = np.minimum(distance, self.max_speed)
        distance_mask = distance > 1e-1
        self.positions[:,distance_mask] += speed[distance_mask] * diff[:,distance_mask] / distance[distance_mask]


def animate_positions(environment, n_frames, interval=100):
    """
    Animates the positions of agents in the environment over time.

    Parameters:
        environment (Environment): The environment object containing agent positions.
        n_frames (int): Number of frames to animate.
        interval (int): Time interval between frames in milliseconds.
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(environment.positions[0], environment.positions[1])

    # Set axis limits (adjust as needed)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Agent Positions Over Time")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    def update(frame):
        for i in range(10):
            environment.update()  # Update the environment
        scatter.set_offsets(environment.positions.T)  # Update scatter plot data
        print(f"Frame {frame} \r", end="")  # Print frame number
        return scatter,

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    plt.show()


def between_goal_calculator(positions, target_agents):

    p1 = positions[:, target_agents[0]]
    p2 = positions[:, target_agents[1]]

    pgoal = 0.5 *(p1 + p2)

    return pgoal

if __name__ == "__main__":
    n_agents = 10
    n_frames = 1000

    env = Environment(n_agents, between_goal_calculator)

    animate_positions(env, n_frames)