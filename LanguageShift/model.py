from LanguageShift.Language import LanguageModel
from LanguageShift.Visualization import Visualization

m = LanguageModel([.009, .013], 'doctoreddata.csv', grid_pickle='neighbor.pkl')
m.run(30)
m.datacollector.get_agent_vars_dataframe().to_csv('output.txt')

v = Visualization(shapefile=[('./shapefiles/AUT_adm2', 'austria'), ('./shapefiles/SVN_adm0', 'slovenia'), ],
                  csv='output.txt')
v.new_figure(title='Locations in our Study')
v.plot_timestep(0)
v.show()