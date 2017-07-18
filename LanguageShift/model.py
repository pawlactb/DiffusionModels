from LanguageShift.Language import LanguageModel

m = LanguageModel([.013, .009], 'doctoreddata.csv', grid_pickle='neighbor.pkl')
m.run(30)
m.datacollector.get_agent_vars_dataframe().to_csv('output.txt')
