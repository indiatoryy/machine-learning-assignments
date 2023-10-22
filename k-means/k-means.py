import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


# Load example dataset
data = load_breast_cancer()
X = data.data
arr = []
arr = X

# K-means algorithm function
def k_means(arr, numCent):
  """
    Perform K-means clustering on the input data.

    Parameters:
    - arr (numpy.ndarray): Input data points.
    - numCent (int): Number of centroids (clusters) to create.

    Returns:
    - List of centroids.
    - List of lists containing data points assigned to each centroid.
    """

  dataDim = len(arr[0]) # dimensionality of data
  centroids = []
  centroidAssig = ([[] for c in range(numCent)])  # initializing list for centroid assignment

  availableCentData = arr
  for e in range (0, numCent):
    randInt = np.random.randint(0, len(availableCentData))
    centroidData = availableCentData[randInt]
    centroids.append(centroidData)

  changeinCent = 1

  # Centroid Assignment
  while changeinCent > 0.00000000000000000000000000001:
    for i in range(0,len(arr)):
      distance = 100000000**10
      closestCent = ([])
      centDistDimensional = 0
      centDist = 0

      for p in range (0,len(centroids)):  # loop centroids to find distance from point
        centDistDimensional = 0
        for d in range (0, dataDim):
            centDistDimensional += (centroids[p][d]-arr[i][d])**2

        centDist = np.sqrt(centDistDimensional)

        if centDist < distance:    # save information if distance is closer
          distance = centDist
          closestCent = centroids[p]
          centNumber = p

      centroidAssig[centNumber].append(arr[i])


    # Reassigning centroids
    numpCentroids = np.array(centroidAssig)

    for c in range(0, numCent):
      changeinCent = 0
      dataAverages = [0 for o in range(dataDim)]

      for d in range (0,dataDim):
        for w in range(0, len(centroidAssig[c])):
          dataAverages[d] += centroidAssig[c][w][d]

        if len(centroidAssig[c]) != 0:
          newCentLocation = dataAverages[d]/len(centroidAssig[c])
          changeinCent += abs(centroids[c][d]-newCentLocation)
          centroids[c][d] = newCentLocation

    return centroids, centroidAssig

# Function to calculate cost for k centroids
def func_cost(numCentroids, centroidAssig, centroids, arr):
  """
    Calculate the cost associated with a set of centroids.

    Parameters:
    - numCentroids (int): Number of centroids.
    - centroidAssig (list): List of lists containing data points assigned to each centroid.
    - centroids (list): List of centroids.
    - arr (numpy.ndarray): Input data points.

    Returns:
    - Cost value.
    """

  sum = 0
  n = len(arr)
  dataDim = len(arr[0])

  for c in range (0, numCentroids):
    centSum = 0

    for i in range (0,len(centroidAssig[c])):

      for d in range(0, dataDim):
        centSum += (centroids[c][d]-centroidAssig[c][i][d])**2
      sum += centSum
  sum /= n
  return sum

# Function to optimize cost by running k-means and cost calculation multiple times
def optimize_cost(numCentroids, arr):
  """
    Run K-means algorithm multiple times to find the optimal centroids and associated cost.

    Parameters:
    - numCentroids (int): Number of centroids.
    - arr (numpy.ndarray): Input data points.

    Returns:
    - Minimum cost value.
    - Centroids associated with the minimum cost.
    - List of lists containing data points assigned to each centroid for the minimum cost.
    """

  cents = []
  assigs = []
  costs = []

  for i in range(0,5):
    cent, assig = k_means(arr, numCentroids)
    cost = func_cost(numCentroids, assig, cent, arr)
    cents.append(cent)
    assigs.append(assig)
    costs.append(cost)

  # determine and return best values
  min_cost = min(costs)
  min_index = costs.index(min_cost)
  min_cents = cents[min_index]
  min_assigs = assigs[min_index]
  return min_cost, min_cents, min_assigs


# Graphing of Results
yVals = []
for i in range (2,8):
  cost, cents, assigs = optimize_cost(i, arr)
  yVals.append(cost)

x= [2,3,4,5,6,7]
y = yVals

plt.plot(x,y,"ob")
plt.title("Number of Centroids vs Distortion")
plt.xlabel("Number of Centrodis")
plt.ylabel("Distortion")
plt.legend('Distortion')
plt.show()