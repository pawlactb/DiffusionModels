from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation

from LanguageShift.NeighborList import NeighborList


class LanguageAgent(Agent):
    def __init__(self, model, unique_id, initial_prob_v):
        """
        A LanguageAgent represents a particular place during a language shift simulation.
        :param model: the model that the agent is in
        :param unique_id: Location number of the agent
        :param initial_prob_v: a list of probabilities of speaking particular languages
        """
        super().__init__(unique_id, model)
        self.probability = np.array(initial_prob_v)
        self.next_probability = np.array(self.probability, copy=True)
        self.diffusion = self.model.diffusion
        self.get_population()

    def get_population(self):
        '''
        Updates the population of the LanguageAgent

        Returns: None

        '''
        self.population = self.model.agent_pop[self.unique_id][self.model.schedule.time]

    def calculate_contribution(self, other):
        '''

        Args:
            other: Another agent for which you want to find the impact from.

        Returns: None

        '''
        ret_val = ((other.population * other.probability) / (4 * np.pi * self.diffusion)) * np.exp(
            -np.square(self.model.grid.get_distance(self, other))) / (4 * self.diffusion)
        return ret_val
        # return ((other.population * other.probability) / (4 * np.pi * self.diffusion)) * np.exp(-np.square(self.model.grid.get_distance(self, other))) / (4 * self.diffusion)

    def step(self):
        '''
        Prepare for the next timestep

        Returns: None

        '''
        f = np.zeros(len(self.probability))
        self.get_population()
        for neighbor in self.model.grid.get_neighbors_by_agent(self)[1:6]:
            f += self.calculate_contribution(neighbor)

        self.next_probability = ((self.population * self.probability) + f) / (np.sum(f) + self.population)

    def advance(self):
        '''
        Advance to the next timestep
        Returns: None

        '''
        self.probability, self.next_probability = self.next_probability, self.probability


# from https://stackoverflow.com/a/4913653/
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def get_distance(a, b):
    '''
    The get_distance function provides the notion of distance for our interactions.

    Args:
        a: the first point
        b: the second point

    Returns: (float) distance between two points in space.

    '''
    return haversine(a.pos[0], a.pos[1], b.pos[0], b.pos[1])


class LanguageModel(Model):
    def __init__(self, diffusivity, filename, grid_pickle=None):
        '''
        LanguageModels contain LanguageAgents and other objects to run the model.

        Args:
            diffusivity:
            filename:
            grid_pickle:
        '''
        super().__init__()
        self.num_agents = 0
        self.grid = NeighborList(distance_fn=get_distance, neighborhood_size=8, loadpickle=grid_pickle)
        self.schedule = SimultaneousActivation(self)
        self.diffusion = np.array(diffusivity)
        self.pop_data = self.read_file(filename)
        self.agent_pop = {}

        # for loc in self.pop_data.loc[:]['location_id']:
        for row in self.pop_data.itertuples(index=True):
            # print('row: ' + str(row))

            # read in population data
            self.agent_pop.update({int(row[1]): [int(x) for x in row[6:]]})
            # print(self.agent_pop[row[1]])

            self.num_agents += 1
            # Create agents, add them to scheduler
            if float(row[11]) == 0:
                a = LanguageAgent(self, int(row[1]), [0, 1])
            else:
                a = LanguageAgent(self, int(row[1]), [float(row[5]), 1 - (float(row[5]))])

            self.schedule.add(a)

            # add the agent at position (x,y)
            # print('lat: ' + str(self.pop_data.loc[idx]['latitude']))
            # print('long ' + str(self.pop_data.loc[idx]['longitude']))
            # print('id:' + str(a.unique_id))
            self.grid.add_agent((float(self.pop_data.loc[a.unique_id - 1]['latitude']),
                                 float(self.pop_data.loc[a.unique_id - 1]['longitude'])), a)
            # print('added')

        if (grid_pickle is None):
            self.grid.calc_neighbors()

        self.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={"pop": lambda x: x.population,
                             "p_slovene": lambda x: x.probability[0],
                             "p_german": lambda x: x.probability[1],
                             "lat": lambda x: x.pos[0],
                             "long": lambda x: x.pos[1],
                             "prob": lambda x: x.probability[0] + x.probability[1]})

    def get_population(self, id, year):
        return self.agent_pop[id][year]

    def read_file(self, filename):
        data = pd.read_csv(filename).dropna()
        # print(data)
        return data

    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()

    def run(self, timesteps):
        for t in range(timesteps):
            print('Model Step: ' + str(self.schedule.time))
            self.step()
