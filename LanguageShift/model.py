from LanguageShift.LanguageModel import LanguageModel
from LanguageShift.Visualization import Visualization

m = LanguageModel(diffusivity=[.009, .013], timestep=1, filename='doctoreddata.csv', grid_pickle=None)
m.run(30)
m.datacollector.get_agent_vars_dataframe().to_csv('output.csv')

v = Visualization(shapefile=[('./shapefiles/AUT_adm2', 'austria'), ('./shapefiles/SVN_adm0', 'slovenia'), ],
                  csv='output.csv')

v.new_figure()
v.plot_timestep(29)
v.show()