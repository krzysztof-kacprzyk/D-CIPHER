import scipy.io
data = scipy.io.loadmat('var_objective/kuramoto_sivishinky.mat')

u = data['uu']
x = data['x'][:,0]
t = data['tt'][0,:]
dt = t[1]-t[0]
dx = x[2]-x[1]

n = len(x)
m = len(t)

print(u.shape)
print(n)
print(m)
print(dt)
print(dx)
print(t[-1])
print(x[-1])