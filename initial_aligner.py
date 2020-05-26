import re
import numpy as np
import networkx as nx
from ase import Atoms, Atom
from ase.io import read, write
from ase import neighborlist
from pymatgen.core import periodic_table
from scipy.spatial import distance_matrix
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from scipy.spatial import distance_matrix
from scipy.optimize import basinhopping, shgo, differential_evolution

def nl(string):
	return re.sub('[^0-9]','', string)

def R(axis, theta):
	"""
		returns a rotation matrix that rotates a vector around axis by angle theta
	"""
	return expm(cross(eye(3), axis/norm(axis)*theta))

def M(vec1, vec2):
	"""
		returns a rotation matrix that rotates vec1 onto vec2
	"""
	ax = np.cross(vec1, vec2)
	
	if np.any(ax): # need to check that the rotation axis has non-zero components
	   
		ax_norm = ax/np.linalg.norm(np.cross(vec1, vec2))
		cos_ang = np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
		ang = np.arccos(cos_ang)
		return R(ax_norm, ang)

	else:
		return eye(3)

def min_dist(coords0, coords1):

	return np.amin(distance_matrix(coords0, coords1))

def maximize_minimum_distance(coords0, coords1, zdist):

	shift_vec = np.array([0.0, 0.0, zdist])

	def objective(X):

		v0, v1, v2, ang = X
		axis = np.array([v0, v1, v2])
		rotM = R(axis, ang)
		result = dot(rotM, coords0.T).T
		md = min_dist(result + shift_vec, coords1)

		return -1*md

	bounds = [(-1.0,1.0), (-1.0,1.0), (-1.0,1.0), (0,2*np.pi)]
	resM = differential_evolution(objective, bounds)

	v0, v1, v2, ang = resM.x
	axis = np.array([v0, v1, v2])
	rotM = R(axis, ang)

	return rotM

def aligner(nodefile, adsfile, zdist=2.5, write_format='xyz'):

	shift_vec = np.array([0.0, 0.0, zdist])
	z = np.array([0.0, 0.0, 1.0])
	
	node = read(nodefile, format=nodefile.split('.')[-1]) 
	adsorbate = read(adsfile, format=adsfile.split('.')[-1])

	# align node metal at (0,0,0) with COM vector along negative z-axis
	metal_coords = np.array([a.position for a in node if periodic_table.Element(a.symbol).is_metal])
	node.positions -= metal_coords[0]
	align_vec = np.average(metal_coords, axis=0)
	rotM = M(align_vec, z)
	node.positions = dot(rotM, node.positions.T).T

	# center adsorbate at (0,0,0) and maximize the minimum distance between the atoms not expected to interact strongly with the metal
	ads_interaction_coords = np.array([a.position for a in adsorbate if a.symbol == 'X'])
	adsorbate.positions -= np.average(ads_interaction_coords, axis=0)
	ads_noninteraction_coords = np.array([a.position for a in adsorbate if a.symbol != 'X'])
	rotM = maximize_minimum_distance(ads_noninteraction_coords, node.positions, zdist)
	adsorbate.positions = dot(rotM, adsorbate.positions.T).T + shift_vec

	# combine node and adsorbate atoms
	all_atoms = Atoms()
	for a in node:
		all_atoms.append(Atom(a.symbol, a.position))
	for a in adsorbate:
		all_atoms.append(Atom(a.symbol, a.position))

	# write all coordinates
	write_name = nodefile.split('.')[0] + '_' + adsfile.split('.')[0] + '.' + write_format 
	write(write_name, all_atoms, format=write_format)

	return all_atoms

aligner('Cu_paddlewheel.xyz', 'CO.xyz')
aligner('Cu_paddlewheel.xyz', 'NH3.xyz')
aligner('Cu_paddlewheel.xyz', 'ethane.xyz')
aligner('Cu_paddlewheel.xyz', 'propene.xyz')
