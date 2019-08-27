import numpy as np
import sklearn.cluster
from astroquery.gaia import Gaia
from astropy.io import fits
from astropy.table import Table

t = Table.read('tst.hdf', format='hdf5')

print t

#r = np.load('area_of_cluster.npy')

#print r

#for i in range (0, 10):

#	print r[i]

