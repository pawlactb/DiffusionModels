import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.basemap import Basemap


class Visualization(object):
    def __init__(self, shapefile, dataFrame=None, csv=None):
        self.shapefile = shapefile

        if dataFrame is None:
            self.data = pd.read_csv(csv)
        else:
            self.data = dataFrame

        if self.data is None:
            raise ValueError('Must pass data to visualization.')

        self.figure_count = 0

    def draw_map(self, res,
                 proj,
                 lat, lon,
                 lllong, lllat, urlong, urlat):
        bmap = Basemap(resolution=res,
                       projection=proj,
                       lat_0=lat, lon_0=lon,
                       llcrnrlon=lllong, llcrnrlat=lllat, urcrnrlon=urlong, urcrnrlat=urlat)

        bmap.drawmapboundary(fill_color='#46bcec')
        bmap.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
        bmap.readshapefile(self.shapefile, 'austria')

    def new_figure(self, title='Figure'):
        # increment the figure count
        self.figure_count += 1

        plt.figure(self.figure_count)

        # plot map with albert's equal area projection
        self.draw_map(res='l',
                      proj='aea',
                      lat=46.64, lon=14.26,
                      lllong=12.5, lllat=46, urlong=15.1, urlat=47.15)

        plt.show()
