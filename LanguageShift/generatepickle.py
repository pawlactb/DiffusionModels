import pickle

from LanguageShift.LanguageModel import LanguageModel

filename = 'neighbor.pkl'
m = LanguageModel([.005, .005], filename='doctoreddata.csv', timestep=1)
pickle.dump(m.grid.agent_neighbors, open(filename, 'wb'))
print('Pickled as ' + filename)
