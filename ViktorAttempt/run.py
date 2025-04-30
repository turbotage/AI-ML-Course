
import numpy as np
import math
#import cupy as cp
try:
	import cupy as cp
	xp = cp
	xp_is_cupy = True
except ImportError:
	print("CuPy not available, using NumPy instead.")
	xp = np
	xp_is_cupy = False

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_numpy(arr):
	return arr.get() if xp_is_cupy else arr


class Environment:
	def __init__(self, n_agents, goal_calculator, target_generator=None, dt=0.1, memory_length=10, perception_radius=0.1):

		self.dt = dt

		self.goal_calculator = goal_calculator

		self.positions = xp.random.rand(2, n_agents).astype(np.float32)
		self.last_positions = xp.copy(self.positions)

		self.max_speed = 1e-2 + xp.abs(1e-2*xp.random.normal(size=(n_agents,)))#xp.ones(n_agents)
		self.perception_radius = perception_radius*xp.ones(n_agents)

		self.memory_target_positions = xp.random.rand(2, 2, n_agents).astype(np.float32)
		self.memory_length = memory_length * xp.ones((n_agents, n_agents), dtype=np.int16)
		
		#self.last_seen_positions = xp.random.rand((2, n_agents, n_agents)).astype(np.float32)

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
		p1 = self.positions[:, self.target_agents[0]]
		p1_dist = xp.linalg.norm(p1 - self.positions, axis=0)
		
		perception_mask = p1_dist > self.perception_radius
		# If the p1 agent is outside the perception radius, we use the last seen position
		p1[:,perception_mask] = self.memory_target_positions[0,:,perception_mask].T

		# If p1 is within perception radius, this is our last seen position
		self.memory_target_positions[0,:,~perception_mask] = p1[:,~perception_mask].T


		p2 = self.positions[:, self.target_agents[1]]
		p2_dist = xp.linalg.norm(p2 - self.positions, axis=0)
		perception_mask = p2_dist > self.perception_radius
		# If the p2 agent is outside the perception radius, we use the last seen position
		p2[:,perception_mask] = self.memory_target_positions[1,:,perception_mask].T

		# If p2 is within perception radius, this is our last seen position
		self.memory_target_positions[1,:,~perception_mask] = p2[:,~perception_mask].T

		# Calculate goal position and move towards it
		self.goal_positions = self.goal_calculator(self.positions, p1, p2)
		
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


def between_goal_calculator(positions, p1, p2, goal_method):
	"""
	Provides the goal calculation for the agents base on the requested method.

	Input:
	:param positions: agent positions
	:param p1: target agent positions 1
	:param p2: target agent positions 2
	:param goal_method: requested goalmethod, either "midpoint", "inbetween", "tailgaiting", "stupid-behind" or "less-stupid-behind"
	"""
	match goal_method:
		case "midpoint":
			pgoal = 0.5 *(p1 + p2)
		case "inbetween":
			# We calculate the projection of the agent position on the line segment p2 + t(p_1 - p_2), t\in [0, 1]
			direction = p1-p2
			norm = xp.linalg.norm(direction, axis=0)
			scalar_projection = xp.linalg.vecdot(positions-p2, direction, axis=0)/norm
			# If we divide by 0 we get NaNs which should be replaced by 0
			scalar_projection[xp.isnan(scalar_projection)] = 0
			t = xp.maximum(xp.minimum(scalar_projection, 1), 0)
			pgoal = p2 + t*direction
		case "tailgating":
			# We calculate the projection of the agent position on the line segment p2 + t(p_1 - p_2), t\in [-\infty, 0]
			direction = p1-p2
			norm = xp.linalg.norm(direction, axis=0)
			scalar_projection = xp.linalg.vecdot(positions-p2, direction, axis=0)/norm
			scalar_projection[xp.isnan(scalar_projection)] = 0
			t = xp.minimum(scalar_projection, 0)
			pgoal = p2 + t*direction
		case "stupid-behind":
			pgoal = p1
		case "less-stupid-behind":
			pgoal = p1 + 0.05*(p1 - p2)
		case _:
			print('''Requested goal_method not implemented. 
					Please use one of "midpoint", "inbetween", "tailgating", 
		 			"stupid-behind" or "less-stupid-behind".''')
	
	return pgoal
	
def target_generator(n_agents, max_n_clusters, singleton_size):

	n_agents_cluster = n_agents - singleton_size
	target_agents, cluster = cluster_generator(n_agents=n_agents_cluster, max_n_clusters=max_n_clusters)
	
		
	target_index_1 = xp.random.randint(low=0, high=n_agents_cluster, size=singleton_size)
	target_index_2 = xp.random.randint(low=0, high=n_agents_cluster, size=singleton_size)

	mask = (target_index_1 == target_index_2)
	target_index_2[mask] = (target_index_2[mask] + 1)%n_agents_cluster

	new_target_agents = xp.stack([target_index_1, target_index_2], axis=0)
	target_agents = xp.concatenate([target_agents, new_target_agents], axis=1)

	cluster = cluster + singleton_size*[cluster[-1]+1]

	return target_agents, cluster

	


def cluster_generator(n_agents, max_n_clusters=10):
	"""
	Generates random target agents for each agent in the environment.

	Parameters:
		n_agents (int): Number of agents.
		n_clusters (int): Number of clusters.

	Returns:
		np.ndarray: Random target agents for each agent.
	"""

	# This method ONLY creates clusters

	if n_agents < 3*max_n_clusters:
		print("We restrict the number of clusters as the number of agents is too small.")
		max_n_clusters = math.floor(n_agents//3)

	def random_partition(n, arr_size):
		"""
		Create a somewhat random integer partition filling the list with 0 if necessary.
		"""
		output = arr_size*[0]
		remain_n = n


		for i in range(arr_size-2):
			value = xp.random.randint(low=remain_n//(i+2), high=remain_n)
			output[i] = value

			remain_n = remain_n - value
		
		output[-1] = remain_n

		return output
	
	# TODO: Maybe add/use binomial distribution

	# This guarantees that our partitions have atleast 3 elements
	randpart = random_partition(n_agents-3*max_n_clusters, max_n_clusters)
	part = [3 + i for i in randpart]

	# # Here we could also use scipy random partition
	# import sympy
	# rand_part = sympy.combinatorics.partitions.random_integer_partition(n_agents-3*max_n_clusters, max_n_clusters)
	# rand_part = rand_part + (max_n_clusters-len(rand_part))*[0]
	# part = max_n_clusters*[3] + rand_part

	# Generate targets which will then approperly rotated.
	target_agents = xp.array((xp.arange(start=0, stop=n_agents, step=1, dtype=int), 
							 xp.arange(start=0, stop=n_agents, step=1, dtype=int)))


	# Start rotating the slices.
	p_end = 0
	for i in range(max_n_clusters):
		p_start = p_end
		p_end = p_end + part[i]
		target_agents[0][p_start:p_end] = xp.roll(target_agents[0][p_start:p_end], 1)
		target_agents[1][p_start:p_end] = xp.roll(target_agents[1][p_start:p_end], -1)

	# This maps the indices to the cluster. Could probably be improved.
	agent_cluster = []
	for i, j in enumerate(part):
		agent_cluster = agent_cluster + j*[i]

	return target_agents, agent_cluster


if __name__ == "__main__":
	n_agents = 500
	timesteps = 12000
	nframes = 6000
	ncluster = 10
	goal_method = "tailgating"

	# target_agents = target_generator(n_agents, 10)

	env = Environment(n_agents, lambda p1, p2, positions: between_goal_calculator(p1=p1, p2=p2, positions=positions, goal_method=goal_method), 
				   lambda n: target_generator(n, ncluster, singleton_size=100), perception_radius=10.0)

	animate_positions(env, timesteps, nframes, interval=1)