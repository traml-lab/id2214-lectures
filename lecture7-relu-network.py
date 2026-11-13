import numpy as np

def g(z): return np.maximum(z, 0) # ReLU

def f1_p(W, b):
    return lambda x : g(W @ x + b)

def f2_p(w, b):
    return lambda x : w @ x + b

# instantiate first function with weights and biases
W1, b1 = np.array([[1.0], [1.0]]), np.array([0.0, -80.0])
f1 = f1_p(W1, b1)

# instantiate second function with weights and biases
w2, b2 = np.array([0.09, 0.05]), 0.3
f2 = f2_p(w2, b2)

# neural network function
def f(x):
    return f2(f1(x))


# Apartment sizes (m²) and their average prices (MSEK)
sizes = np.array([22.5, 41.9, 61.8, 81.7, 101.6, 121.6, 152.3])
prices = np.array([2.635, 4.017, 6.052, 7.989, 10.702, 13.283, 17.617])

predictions = np.zeros(len(sizes))
for n in range(len(sizes)):
    x = np.array([sizes[n]])
    h = f1(x)
    predictions[n] = f2(h)
    print(f"{sizes[n]:5.1f} m²:  h = [{h[0]:5.1f}, {h[1]:4.1f}]   "
          f"f(x) = {predictions[n]:5.2f}   y = {prices[n]:5.2f} MSEK")

print(f"Squared loss: {np.sum((prices - predictions) ** 2):.2f}")

# Predict the price of a new apartment
print(f"A 125 m² apartment costs about {f(np.array([125.0])):.2f} MSEK")
