import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace( mu - 3 *sigma, mu + 3*sigma, 1000 )

# stats.norm.pdf - API:
# https://scipy.github.io/devdocs/generated/scipy.stats.norm.html?highlight=pdf
plt.plot(x, stats.norm.pdf(x, loc=mu, scale=sigma))


# plt.hist(stats.norm.pdf(x, mu, sigma), bins=20)
plt.show()

print(x)

# print(stats.norm.pdf(x, mu, sigma))