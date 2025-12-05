import openpnm as op
import matplotlib.pyplot as plt
import numpy as np

op.visualization.set_mpl_style()
np.random.seed(10)
np.set_printoptions(precision=5)

#Create a cubic network
shape = [10, 10, 1]
spacing = 1e-5
net = op.network.Cubic(shape=shape, spacing=spacing)

#Add geometry models
geo = op.models.collections.geometry.spheres_and_cylinders
net.add_model_collection(geo, domain='all')
net.regenerate_models()

# Define phase
# Use a simple Phase instead of the Air helper to avoid loading
# external thermo datasets. Set a constant pore diffusivity.
air = op.phase.Phase(network=net)
air['pore.diffusivity'] = np.ones(net.Np) * 1e-5

# Add basic physics models but remove any model that may require
# external thermo data
phys = op.models.collections.physics.basic
if 'throat.entry_pressure' in phys:
	del phys['throat.entry_pressure']
air.add_model_collection(phys)
air.regenerate_models()

#perform diffusion simulation
fd = op.algorithms.FickianDiffusion(network=net, phase=air)

def _boundary_pores_by_coord(network, axis=0):
	coords = network['pore.coords']
	mn = coords[:, axis].min()
	mx = coords[:, axis].max()
	inlet = np.where(np.isclose(coords[:, axis], mn))[0]
	outlet = np.where(np.isclose(coords[:, axis], mx))[0]
	return inlet, outlet

# Try labeled faces first, else fall back to coordinates
try:
	inlet = net.pores('left')
	outlet = net.pores('right')
except KeyError:
	inlet, outlet = _boundary_pores_by_coord(net, axis=0)

C_in, C_out = [10, 5]
fd.set_value_BC(pores=inlet, values=C_in)
fd.set_value_BC(pores=outlet, values=C_out)

fd.run();

#visualize results
pc = fd['pore.concentration']
tc = fd.interpolate_data(propname='throat.concentration')
d = net['pore.diameter']
fig, ax = plt.subplots(figsize=[5, 5])
op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=400, ax=ax)
op.visualization.plot_connections(network=net, color_by=tc, linewidth=3, ax=ax)
_ = plt.axis('off')
plt.title('Fickian Diffusion')
# Save figure instead of blocking with plt.show() to allow scripts to
# run non-interactively. File will be created in the current folder.
fig.savefig('fickian_diffusion.png', dpi=150, bbox_inches='tight')
plt.close(fig)

#effective diffusivity
rate_inlet = fd.rate(pores=inlet)[0]
print(f'Molar flow rate: {rate_inlet:.5e} mol/s')

A = (shape[1] * shape[2])*(spacing**2)
L = shape[0]*spacing
D_eff = rate_inlet * L / (A * (C_in - C_out))
print("{0:.6E}".format(D_eff))

#tortuosity
V_p = net['pore.volume'].sum()
V_t = net['throat.volume'].sum()
V_bulk = np.prod(shape)*(spacing**3)
e = (V_p + V_t) / V_bulk
print('The porosity is: ', "{0:.6E}".format(e))

D_AB = air['pore.diffusivity'][0]
tau = e * D_AB / D_eff
print('The tortuosity is:', "{0:.6E}".format(tau))