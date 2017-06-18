from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation
import numpy as np
import pandas as pd
import random
from AdjacencyGrid import AdjacencyGrid
import pickle

class LanguageAgent(Agent):
    def __init__(self, model, unique_id, initial_population, initial_prob_v):
        super().__init__(unique_id, model)
        self.population = initial_population
        self.probability = np.array(initial_prob_v)
        self.next_probability = np.array(self.probability, copy=True)
        self.diffusion = self.model.diffusion

    def calculate_contribution(self, other):
        '''
        args:
            other(LanguageAgent): an adjacent or otherwise relevant other LanguageAgent
        '''
        return ((other.population * other.probability) / (4 * np.pi * self.diffusion)) * np.exp(
            -np.square(self.model.grid.get_distance(self.pos, other.pos))) / (4 * self.diffusion)

    def step(self):
        f = np.zeros(len(self.probability))
        for neighbor in self.model.grid.get_neighbors_by_agent(self)[1:8]:
            f += self.calculate_contribution(neighbor)

        self.next_probability = ((self.population * self.probability) + f) / (np.sum(f) + self.population)

    def advance(self):
        self.probability, self.next_probability = self.next_probability, self.probability


class LanguageModel(Model):
    def __init__(self, diffusivity, filename):
        super()
        self.num_agents = 0
        #self.grid = ContinuousSpace(1000, 1000, torus=False)
        self.grid = AdjacencyGrid()
        self.schedule = SimultaneousActivation(self)
        self.diffusion = np.array(diffusivity)
        self.pop_data = self.read_file(filename)

        for loc in self.pop_data.loc[:]['location_id']:
                #print('id: ' + str(idx))
                self.num_agents += 1
                # Create agents, add them to scheduler
                a = LanguageAgent(self, loc, 1000 + 500 * random.random(), [random.random(), random.random()])
                self.schedule.add(a)

                # add the agent at position (x,y)
                #print('lat: ' + str(self.pop_data.loc[idx]['latitude']))
                #print('long ' + str(self.pop_data.loc[idx]['longitude']))
                self.grid.place_agent(a, (float(self.pop_data.loc[a.unique_id - 1]['latitude']), float(self.pop_data.loc[a.unique_id - 1]['longitude'])))
                #print('added')

        self.grid.calc_neighbors()

        self.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={"diff": lambda x: x.next_probability - x.probability})

    def read_file(self, filename):
        data = pd.read_csv(filename)
        return data.dropna()


    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()

    def run(self, timesteps):
        for t in range(timesteps):
            self.step()
            print('Model Step: ' + str(t+1))


m = LanguageModel([.05, .05], 'speakers.csv')
pickle.dump(m, file=open( "languagemodel.p", "wb" ))

