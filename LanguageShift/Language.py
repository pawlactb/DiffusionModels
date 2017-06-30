import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation
from scipy.spatial import distance

from LanguageShift.NeighborList import NeighborList


class LanguageAgent(Agent):
    def __init__(self, model, unique_id, intitial_population, initial_prob_v):
        super().__init__(unique_id, model)
        self.probability = np.array(initial_prob_v)
        self.next_probability = np.array(self.probability, copy=True)
        self.diffusion = self.model.diffusion
        self.population = self.get_population()

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
        super().__init__()
        self.num_agents = 0
        self.grid = NeighborList(distance_fn=get_distance, neighborhood_size=8)
        self.schedule = SimultaneousActivation(self)
        self.diffusion = np.array(diffusivity)
        self.pop_data = self.read_file(filename)
        self.agent_pop = {}

        # for loc in self.pop_data.loc[:]['location_id']:
        for row in self.pop_data.itertuples(index=True):
            print('id: ' + str(row))
            self.num_agents += 1
            # Create agents, add them to scheduler
            a = LanguageAgent(self, int(row[0]), [int(row[11]), int(row[12]), int(row[13]), int(row[14])],
                              [float(row[15] / float(row[11])), 1 - float(row[15] / float(row[11]))])
            self.schedule.add(a)

            # add the agent at position (x,y)
            # print('lat: ' + str(self.pop_data.loc[idx]['latitude']))
            # print('long ' + str(self.pop_data.loc[idx]['longitude']))
            self.grid.add_agent((float(self.pop_data.loc[a.unique_id - 1]['latitude']),
                                 float(self.pop_data.loc[a.unique_id - 1]['longitude'])), a)
            # print('added')

        # self.grid.calc_neighbors()

        self.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={"pop": lambda x: x.population * x.probability[0]})

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
