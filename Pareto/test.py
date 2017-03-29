#Let's have a go at doing a Gaussian process ??RBF??
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

""" This is code for simple GP regression. It assumes a zero mean GP Prior """
f = lambda x: np.sin(0.01*x).flatten()
# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.01
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)



#one of the parameters from Pareto front
#X_non_dom = [0.01, 0.06699999999999999, 0.047999999999999994, 0.143, 0.12399999999999997, 0.028999999999999998, 0.08599999999999998]

#Coefficient of lift
X_non_dom = [-0.6498, -0.6292, -0.6427, -0.5559, -0.5758, -0.6492, -0.6125]
#Coefficient of drag
Y_non_dom = [0.24807, 0.23502, 0.23736, 0.23364, 0.23402, 0.24235, 0.23471]




N = len(X_non_dom)       # number of training points.
n = 50         # number of test points.
s = 0.00005    # noise variance.


# Let's import the Pareto Front points
X = (np.asarray(X_non_dom))
X.shape=((len(X_non_dom)),1)
y = Y_non_dom



K = kernel(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))

# points we're going to make predictions at.
Xtest = np.linspace(-1.5, 0, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)


# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=2)
#pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.plot(X, y, 'b.', markersize=10, label=u'Observations')
#pl.savefig('predictive.png', bbox_inches='tight')
#pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-1, 0, 0.01,0.4]) #pl.axis([-0.7, -0.5, 0.22,0.25])
'''
# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,10)))
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('Ten samples from the GP prior')
#pl.axis([-1.5, 0.5, 0,2])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('Ten samples from the GP posterior')
#pl.axis([-2, 0, 0,2])
pl.savefig('post.png', bbox_inches='tight')
'''
pl.show()


#acquisition function
#def acquisitionfunction()