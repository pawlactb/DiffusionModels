import heapq
import itertools
import pickle

import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation
from scipy.spatial import distance
from scipy import interpolate

class heap(object):
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def heapify(self):
        heapq.heapify(self.pq)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        'Remove and return the highest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def get_smallest(self, n):
        self.heapify()
        return heapq.nsmallest(n, self.pq)[:]



class NeighborList(object):
    """ Grid where adjacency is stored, rather calculated. Used for fast neighbor lookups.


    Performance:
        Let n be number of Agents

        Obtain agent by ID:
            O(n) (worst case)
        Obtain neighborhood of agent:
            O(n) (worst case)
        Calculate neighborhood (performed once at beginning):
            O((n**2)log_2(n)) (worst case)
            O(n**2 + n) (average case for small neighborhoods)


    Methods:
        get_neighbors: Returns the objects surrounding a given cell.
    """

    def __init__(self, distance_fn, neighborhood_size, loadpickle=None):
        self.agent_list = {}
        if loadpickle is not None:
            self.agent_neighbors = pickle.load(open(loadpickle, 'rb'))
        else:
            self.agent_neighbors = {}
        self.get_distance = distance_fn
        self.neighborhood_size = neighborhood_size

    def calc_neighbors(self):
        '''
        After adding all agents to the model, calculate the neighbors


        '''

        # Store other agents in a priority queue by distance (lesser distance == higher priority)
        # We pop a tuple of type (int, agent). The first element is the agent id.

        # iterate over all agents
        for a_id, a in self.agent_list.items():
            # create a priority queue
            neighbors_pq = heap()
            # print('Agent #' + str(a.unique_id))

            #iterate through all agents to calculate the distance to all other agents
            for b_id, b in self.agent_list.items():
                # print('\tadded: #' + str(b.unique_id))
                #add the agent's id to the priority queue, with the distance as the priority
                neighbors_pq.add_task(b_id, priority=self.get_distance(a, b))
            neighbors_pq.heapify()
            neighbors = [i[2] for i in neighbors_pq.get_smallest(self.neighborhood_size + 1)]
            # print('\t\tadding ' + str(neighbors))
            self.agent_neighbors.update({a_id: neighbors})
            del neighbors_pq, neighbors
            #self.agent_neighbors[a] = sorted(neighbors, key=lambda b: self.get_distance(a, b))

    def add_agent(self, pos, agent):
        x, y = pos
        agent.pos = pos
        self.agent_list[agent.unique_id] = agent

    def get_agent(self, a):
        return self.agent_list[a.unique_id]

    def get_neighbors_by_agent(self, agent, include_self=False):
        if include_self:
            return [self.agent_list[a_id] for a_id in self.agent_neighbors[agent.unique_id]]

        ret_val = [self.agent_list[a_id] for a_id in self.agent_neighbors[agent.unique_id]][1:]
        # print(ret_val, type(ret_val))
        return ret_val


class LanguageAgent(Agent):
    def __init__(self, model, unique_id, population_v, initial_prob_v):
        super().__init__(unique_id, model)
        print('pop_v: ' + str(population_v))
        self.population_v = np.arange(0, 30, 1)
        self.get_population = interpolate.interp1d([0, 10, 20, 30], population_v, kind='linear')
        print('inter_p: ' + str(self.get_population(self.population_v)))
        self.probability = np.array(initial_prob_v)
        self.next_probability = np.array(self.probability, copy=True)
        self.diffusion = self.model.diffusion
        self.population = self.get_population(0)

    def get_population(self):
        self.population = self.model.agent_pop[self.unique_id][self.model.schedule.time]

    def calculate_contribution(self, other):
        '''
        args:
            other(LanguageAgent): an adjacent or otherwise relevant other LanguageAgent
        '''
        return ((other.population * other.probability) / (4 * np.pi * self.diffusion)) * np.exp(
            -np.square(self.model.grid.get_distance(self, other))) / (4 * self.diffusion)

    def step(self):
        f = np.zeros(len(self.probability))
        self.population = self.get_population()
        for neighbor in self.model.grid.get_neighbors_by_agent(self)[1:8]:
            f += self.calculate_contribution(neighbor)

        self.next_probability = ((self.population * self.probability) + f) / (np.sum(f) + self.population)

    def advance(self):
        self.probability, self.next_probability = self.next_probability, self.probability


def get_distance(a, b):
    return distance.euclidean(a.pos, b.pos)

class LanguageModel(Model):
    def __init__(self, diffusivity, filename):
        super()
        self.num_agents = 0
        self.grid = NeighborList(distance_fn=get_distance, neighborhood_size=8, loadpickle='neighbor.pkl')
        self.schedule = SimultaneousActivation(self)
        self.diffusion = np.array(diffusivity)
        self.pop_data = self.read_file(filename)

        #for loc in self.pop_data.loc[:]['location_id']:
        for row in self.pop_data.itertuples(index=True):
            print('id: ' + str(row))
            self.num_agents += 1
            # Create agents, add them to scheduler
            a = LanguageAgent(self, int(row[0]), [int(row[11]), int(row[12]), int(row[13]), int(row[14])], [float(row[15]/float(row[11])), 1-float(row[15]/float(row[11]))])
            self.schedule.add(a)

            # add the agent at position (x,y)
            #print('lat: ' + str(self.pop_data.loc[idx]['latitude']))
            #print('long ' + str(self.pop_data.loc[idx]['longitude']))
            self.grid.add_agent((float(self.pop_data.loc[a.unique_id - 1]['latitude']),
                                     float(self.pop_data.loc[a.unique_id - 1]['longitude'])), a)
                #print('added')


        #self.grid.calc_neighbors()

        self.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={"pop": lambda x:  x.population*x.probability[0]})

    def read_file(self, filename):
        data = pd.read_csv(filename)
        print(data)
        return data


    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()

    def run(self, timesteps):
        for t in range(timesteps):
            self.step()
            print('Model Step: ' + str(self.schedule.time))


m = LanguageModel([.005, .005], 'doctoreddata.csv')
m.run(30)
print(m.datacollector.get_agent_vars_dataframe().tail())
