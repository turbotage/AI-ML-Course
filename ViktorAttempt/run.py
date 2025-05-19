
import numpy as np
import math
import networkx as nx
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

class EnvironmentSettings:
	def __init__(self):
		self.n_agents = 100
		self.goal_calculator = lambda p1, p2, positions: goal_calculator(p1=p1, p2=p2, positions=positions, goal_method="midpoint")
		self.target_generator = None
		self.dt = 0.01
		self.mean_perception_radius = 0.1
		self.std_perception_radius = 0.1

		self.mean_communication_radius = 0.1
		self.std_communication_radius = 0.1

		self.mean_speed = 1e-2
		self.std_speed = 1e-2
		

class Environment:
	def __init__(self, envsettings: EnvironmentSettings):

		self.dt = envsettings.dt

		self.goal_calculator = envsettings.goal_calculator

		self.n_agents = n_agents

		self.positions = xp.random.rand(2, n_agents).astype(np.float32)
		self.last_positions = xp.copy(self.positions)

		self.speed = 1e-2 + xp.abs(1e-2*xp.random.normal(size=(n_agents,)))#xp.ones(n_agents)
		self.speed = envsettings.mean_speed + xp.abs(xp.random.normal(scale=envsettings.std_speed, size=(n_agents,)))


		self.perception_radius = envsettings.mean_perception_radius + xp.abs(xp.random.normal(scale=envsettings.std_perception_radius, size=(n_agents,)))
		
		
		self.communication_radius = envsettings.mean_communication_radius + xp.abs(xp.random.normal(scale=envsettings.std_communication_radius, size=(n_agents,)))

		self.memory_positions = 0.45 + 0.1*xp.random.rand(2, n_agents, n_agents).astype(np.float32)
		self.memory_length = xp.ones((n_agents, n_agents), dtype=np.int16)
		
		#self.last_seen_positions = xp.random.rand((2, n_agents, n_agents)).astype(np.float32)

		#self.target_agents = np.zeros((2, n_agents), dtype=np.int)
		#self.target_agents = np.random.randint(0, n_agents-1, size=(2, n_agents))
		self.cluster_indices = xp.zeros((n_agents), dtype=np.int32)

		if envsettings.target_generator is not None:
			self.target_agents, self.agent_clusters = envsettings.target_generator(n_agents)
		else:
			self.target_agents = xp.random.randint(0, n_agents, size=(2, n_agents))
			self.agent_clusters = None
			
		self.goal_positions = xp.random.rand(2, n_agents)

	def update(self):
		
		# Update memory length, it has been one iteration since we last saw any other agent
		self.memory_length += 1

		# Copy last updates positions and velocities
		self.last_positions = xp.copy(self.positions)


		p1 = xp.zeros((2, self.n_agents), dtype=np.float32)
		p2 = xp.zeros((2, self.n_agents), dtype=np.float32)

		# Calculate pairwise distances between agents
		position_array = self.positions[...,None].repeat(self.n_agents, axis=2)
		within_perception_radius = xp.linalg.norm(self.positions[...,None] - position_array.transpose(0,2,1), axis=0)

		# Check which agents are within the communication radius
		communication_mask = within_perception_radius < self.communication_radius
		# Check which agents are within the perception radius
		perception_mask = within_perception_radius < self.perception_radius

		# Update the memory positions of the agents within the perception radius
		self.memory_positions[:, perception_mask] = position_array[:, perception_mask]
		# Agents should have zero memory length of agents within their perception radius
		self.memory_length[perception_mask] = 0

		# Loop over each agent, setting their target positions based on the last memory of the target agents
		for n in range(self.n_agents):
			commask = communication_mask[n,:]
			idx = xp.where(commask)[0]
			
			last_mem_idx = idx[xp.argmin(self.memory_length[commask, self.target_agents[0, n]])]
			# idx[last_mem_idx] holds the index of the agent within the communication radius of the n-th agent
			# with the shortest memory length to the target, if no other agent is within the communication radius
			# this is the n-th agent itself

			p1[:,n] = self.memory_positions[:, last_mem_idx, self.target_agents[0, n]]

			last_mem_idx = idx[xp.argmin(self.memory_length[commask, self.target_agents[1, n]])]
			p2[:,n] = self.memory_positions[:, last_mem_idx, self.target_agents[1, n]]

		# Calculate goal position and move towards it
		self.goal_positions = self.goal_calculator(self.positions, p1, p2)
		
		diff = self.goal_positions - self.positions
		distance = xp.linalg.norm(diff, axis=0)

		vdt = self.speed * self.dt
		movement_mask = distance > vdt
		self.positions[:,movement_mask] += vdt[movement_mask] * diff[:,movement_mask] / distance[movement_mask]
		self.positions[:,~movement_mask] = self.goal_positions[:,~movement_mask]

		# Walls
		self.positions = xp.clip(self.positions, 0, 1)

	def get_average_velocity(self):
		return xp.linalg.norm(self.positions - self.last_positions, axis=0).mean() / self.dt

	#def has_converged(self):
		


def animate_positions(environment: Environment, timesteps, nframes, interval=100, filename="agent_animation.gif", save=False):
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
	# plt.show()
	if save:
		anim.save(filename + "_gif.gif", fps=15) # JGA: Tried 30 and 10...
		print("Saving gif complete")
	else:
		plt.show()
	avg_vel = xp.array(avg_vel)
	
	plt.figure()
	plt.plot(get_numpy(avg_vel))
	plt.title("Average Velocity Over Time")

	if save:
		plt.savefig(filename + "_plt.jpg")
	else:
		plt.show()

	plt.figure()
	create_graph_repr(env.target_agents)
	plt.plot()

	if save:
		plt.savefig(filename + "_graph.jpg")
	else:
		plt.show()

def goal_calculator(positions, p1, p2, goal_method):
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

def random_generator(n_agents):
	"""
	Creates random admissible targets
	"""
	# This is our identiy vector
	id = xp.arange(start=0, stop=n_agents)
	# Initialize an random array with values between 1 and n-1
	rand_arr_1 = xp.random.randint(low=1, high=n_agents, size=n_agents)
	# Adding the identity to the random arry and taking modulo n gives a function which has no fixed points.
	target_agents_1 = (rand_arr_1 + id)%n_agents
	# With same idea as above, generate a random array which has no duplicates as in rand_arr_1
	# Here we use the argument with n-1 instead of n as we need to have it different to our id.
	# Furthermore, we add 1 at the end to put it in the correct range.
	rand_arr_2 = (rand_arr_1 + xp.random.randint(low=0, high=n_agents-2, size=n_agents))%(n_agents-1) + 1
	target_agents_2 = (rand_arr_2 + id)%n_agents

	target_agents = xp.array((target_agents_1, target_agents_2))
	agent_cluster = list(range(n_agents)) #n_agents*[0]

	return target_agents, agent_cluster

def create_graph_repr(target_agents):
	target_agents_1, target_agents_2 = target_agents[0], target_agents[1]
	edge_list_1 = [(ind, targ_ind) for ind, targ_ind in enumerate(target_agents_1)]
	edge_list_2 = [(ind, targ_ind) for ind, targ_ind in enumerate(target_agents_2)]

	G = nx.DiGraph()
	G.add_edges_from(edge_list_1, color="r")
	G.add_edges_from(edge_list_2, color="b")

	pos = nx.spring_layout(G, seed=13648)
	nx.draw_networkx_nodes(G, pos, node_size=10)
	nx.draw_networkx_edges(G, pos, edgelist=edge_list_1, edge_color="skyblue")
	nx.draw_networkx_edges(G, pos, edgelist=edge_list_2, edge_color="blueviolet")

	return G


if __name__ == "__main__":
	n_agents = 200
	timesteps = 1200
	nframes = timesteps // 2
	ncluster = 15
	goal_method = "inbetween"
	
	perception_radius = 0.1 # Is this used somewhere?

	from datetime import datetime

	for i in range(5):
		# TODO: JGA: I hate this
		envsettings = EnvironmentSettings()
		envsettings.n_agents = n_agents
		envsettings.goal_calculator = lambda p1, p2, positions: goal_calculator(p1=p1, p2=p2, positions=positions, goal_method=goal_method)
		envsettings.target_generator = lambda n: random_generator(n)
		envsettings.mean_perception_radius = 0.1
		envsettings.std_perception_radius = 0.0
		envsettings.mean_communication_radius = 0.000001 # envsettings.mean_perception_radius
		envsettings.std_communication_radius = 0.0
		envsettings.mean_speed = 1e-2
		envsettings.std_speed = 0.0


		env = Environment(envsettings)

		filename = f"saved_gifs//{datetime.today().strftime('%Y-%m-%d')}_{goal_method}_nagents_{n_agents}_"
		filename += f"mu_pr_{envsettings.mean_perception_radius}_std_pr_{envsettings.std_perception_radius}_"
		filename += f"mu_cr_{envsettings.mean_communication_radius}_std_cr_{envsettings.std_communication_radius}_"
		filename += f"mu_speed_{envsettings.mean_speed}_std_speed_{envsettings.std_speed}_"
		filename += f"ndt_{timesteps}_nf_{nframes}_{i}"

		animate_positions(env, timesteps, nframes, interval=0, filename=filename, save=True)