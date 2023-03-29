import numpy as np
import matplotlib.pyplot as plt

# Minimise material cost of manufacturing a cylindrical barrel
# of fixed volume V.

# Cost function: surface area of cylinder
def S(r, V):
    return 2 * np.pi * r**2 + (2 * V) / r


# Apply Newton's method to derivative
def G(r, V):
    '''
    r[k+1] = r[k] - S'(r[k]) / S''(r[k])
    
    r[k+1] - r[k] = S'(r[k]) / S''(r[k])
    '''
    Sp = 4 * np.pi * r - (2 * V) / r**2
    Spp = 4 * np.pi + (4 * V) / r**3
    
    return r - Sp / Spp


# Plot the function
V = 1
r_values = np.linspace(0.4, 0.8, 200)

fig, ax = plt.subplots()
ax.plot(r_values, S(r_values, V))
ax.set(title=f'Surface area of a barrel of volume {V} m^3',
       xlabel='Radius (m)', ylabel='Surface area (m^2)')
    #    ylim=[5, 10])
# plt.show()

# Set up Newton's method
r = 0.2
tol = 1e-4
dist = np.inf
dist_values = []

# Iterate as long as we are still improving the guess significantly
while dist > tol:
    # Newton iteration
    r_new = G(r, V)
    
    # Calculate distance between guesses
    dist = abs(r_new - r)
    dist_values.append(dist)
    
    # Prepare next iteration
    r = r_new

# Display the minimum we found
ax.plot(r, S(r, V), 'rx')
# plt.show()

print(f'The material cost is minimised for r = {r:.5g}, and h = {V / (np.pi * r**2):.5g}')

r_exact = (V / (2 * np.pi)) ** (1/3)
print(f'r = {r:.15g}\nr_exact = {r_exact:.15g}')
print(f'The difference between exact and approximated solutions is {abs(r - r_exact):.5g}')

# Plot the successive distances between guesses
fig, ax = plt.subplots()
ax.plot(dist_values, 'bx')
ax.set(xlabel='Iteration number', ylabel='Distance from previous guess', xscale='log', yscale='log')
plt.show()