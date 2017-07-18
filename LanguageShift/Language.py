import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation
from scipy.spatial import distance

from LanguageShift.NeighborList import NeighborList


class LanguageAgent(Agent):
    def __init__(self, model, unique_id, initial_prob_v):
        """

        :param model:
        :param unique_id:
        :param initial_prob_v:
        """
        super().__init__(unique_id, model)
        self.probability = np.array(initial_prob_v)
        self.next_probability = np.array(self.probability, copy=True)
        self.diffusion = self.model.diffusion
        self.get_population()

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
        print(self.population, self.probability)
        f = np.zeros(len(self.probability))
        self.get_population()
        for neighbor in self.model.grid.get_neighbors_by_agent(self)[1:8]:
            f += self.calculate_contribution(neighbor)

        self.next_probability = ((self.population * self.probability) + f) / (np.sum(f) + self.population)

    def advance(self):
        self.probability, self.next_probability = self.next_probability, self.probability


def get_distance(a, b):
    return distance.euclidean(a.pos, b.pos)


class LanguageModel(Model):
    def __init__(self, diffusivity, filename, grid_pickle=None):
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
            self.agent_pop.update({int(row[1]): [int(x) for x in row[5:]]})
            # print(self.agent_pop[row[1]])

            self.num_agents += 1
            # Create agents, add them to scheduler
            if float(row[11]) == 0:
                a = LanguageAgent(self, int(row[1]), [0, 1])
            else:
                a = LanguageAgent(self, int(row[1]),
                                  [float(row[15]) / float(row[11]), 1 - (float(row[15]) / float(row[11]))] )

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
            agent_reporters={"pop": lambda x: x.population * x.probability[0]})

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
