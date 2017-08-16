import numpy as np
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import SimultaneousActivation

from LanguageShift.LanguageAgent import LanguageAgent
from LanguageShift.NeighborList import NeighborList


class LanguageModel(Model):
    def __init__(self, diffusivity, timestep, filename, grid_pickle=None):
        '''
        LanguageModels contain LanguageAgents and other objects to run the model.
        Args:
            diffusivity:
            filename:
            grid_pickle:
        '''
        super().__init__()
        self.num_agents = 0
        self.grid = NeighborList(neighborhood_size=8, loadpickle=grid_pickle)

        self.schedule = SimultaneousActivation(self)
        self.diffusion = np.array(diffusivity)
        self.pop_data = self.read_file(filename)
        self.agent_pop = {}
        self.timestep = timestep

        # for loc in self.pop_data.loc[:]['location_id']:
        for row in self.pop_data.itertuples(index=True):
            # print('row: ' + str(row))

            # read in population data
            self.agent_pop.update({int(row[1]): [int(x) for x in row[6:]]})
            # print(self.agent_pop[row[1]])

            self.num_agents += 1
            # Create agents, add them to scheduler
            if float(row[11]) == 0:
                a = LanguageAgent(self, str(row[2]), int(row[1]), [0, 1])
            else:
                a = LanguageAgent(self, str(row[2]), int(row[1]), [float(row[5]), 1 - (float(row[5]))])

            self.schedule.add(a)

            # add the agent at position (x,y)
            # print('lat: ' + str(self.pop_data.loc[idx]['latitude']))
            # print('long ' + str(self.pop_data.loc[idx]['longitude']))
            print('id:' + str(a.unique_id))
            self.grid.add_agent((float(self.pop_data.loc[a.unique_id - 1]['latitude']),
                                 float(self.pop_data.loc[a.unique_id - 1]['longitude'])), a)
            # print('added')

        if grid_pickle is None:
            self.grid.calc_neighbors()

        self.datacollector = DataCollector(
            model_reporters={},
            agent_reporters={"pop": lambda x: x.population,
                             "p_p_german": lambda x: x.p_probability[0],
                             "p_p_slovene": lambda x: x.p_probability[1],
                             "p_german": lambda x: x.probability[0],
                             "p_slovene": lambda x: x.probability[1],
                             "p_diff": lambda x: np.sum(np.abs(x.probability - x.p_probability)),
                             "lat": lambda x: x.pos[0],
                             "long": lambda x: x.pos[1]})

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
