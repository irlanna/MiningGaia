import sklearn.cluster
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import sys
import itertools
from scipy import linalg


def plot_results(means, covariances, fig, ax):
    index = 0
    #splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar) in enumerate(zip(
            means, covariances)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        #if not np.any(Y_ == i):
        #    continue
        #plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

	print '-->', mean, v[0], v[1]

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, edgecolor='red', visible = True, facecolor='none')
        ell.set_clip_box(ax.bbox)
        #ell.set_alpha(99.0)
        ax.add_artist(ell)

    #plt.xlim(-9., 5.)
    #plt.ylim(-3., 6.)
    #plt.xticks(())
    #plt.yticks(())
    #plt.title(title)


t = Table.read('tst_18maglimit.hdf', format='hdf5')


#N = 100000
X = np.array([t['ra'], t['dec'], t['parallax'], t['pmra'], t['pmdec'] ])
#X = np.array([t['ra'][0:N], t['dec'][0:N], t['parallax'][0:N], t['pmra'][0:N], t['pmdec'][0:N] ])

Xnew = np.transpose(X)

print 'Maximum and minimum parallax in the sample: ', np.max(t['parallax']), np.min(t['parallax'])

f = open('clusters.txt', 'r')

cn = 0

ra = []
dec = []

for lines in f.readlines():

	line = lines.split()

	#print line

	if cn > 2:
	
		ra.append(float(line[1]))
		dec.append(float(line[2]))	

	cn =cn + 1


print ra

#sys.exit(0)

clustering = DBSCAN(eps=0.02, min_samples=2, leaf_size=30).fit(Xnew)
#gmm = GaussianMixture(n_components=200).fit(Xnew)
labels = clustering.labels_

fig, axes =  plt.subplots ()

plt.scatter(Xnew[:, 0], Xnew[:, 1], c=labels, s=1, cmap='viridis')
plt.scatter(ra, dec, s=36, color ='red', marker='^')
plt.xlabel('RA')
plt.ylabel('DEC')
plt.show()

#gmm_means = []
#gmm_cov = []

#for i in range (0, len(gmm.means_)):

#	print i, gmm.covariances_[i][0][0]
	
#	gmm_means.append([gmm.means_[i][0], gmm.means_[i][1]])
#	gmm_cov.append([[gmm.covariances_[i][0][0], gmm.covariances_[i][0][1]], [gmm.covariances_[i][1][0], gmm.covariances_[i][1][1]]  ])


#plot_results(gmm_means, gmm_cov, fig, axes)

#plt.show()

#print gmm.get_params()



#labels_1 = gmm.fit_predict(Xnew)


#plt.scatter(Xnew[:, 0], Xnew[:, 1], c=labels_1, s=7, cmap='viridis')

#labels_2 = gmm.get_params(deep=True)
#print (labels_2)


