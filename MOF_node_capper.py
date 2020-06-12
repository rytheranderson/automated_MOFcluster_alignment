import re
import glob
import networkx as nx
import numpy as np
from datetime import datetime
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from ase.io import read, write
from ase import neighborlist, Atom
from itertools import chain

def flatten(L):

	return list(chain(*L))

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

def atoms2graph(atoms):

	unit_cell = atoms.get_cell()
	cutoffs = neighborlist.natural_cutoffs(atoms)
	NL = neighborlist.NewPrimitiveNeighborList(cutoffs, self_interaction=False, skin=0.1) # default atom cutoffs work well
	NL.build([False, False, False], unit_cell, atoms.positions)

	G = nx.Graph()

	for a, ccoord in zip(atoms, atoms.positions):
		ind = a.index + 1
		G.add_node(a.symbol + str(ind), element_symbol=a.symbol, ccoord=ccoord, fcoord=np.array([0.0,0.0,0.0]), freeze_code='0')

	for a in G.nodes():
		
		nbors = [atoms[i].symbol + str(atoms[i].index+1) for i in NL.get_neighbors(int(nl(a))-1)[0]]
		
		for nbor in nbors:
			G.add_edge(a, nbor, bond_type='', bond_sym='.', bond_length=0)

	return G

def cap_node(filename, cap_elems, cap_vecs, bond_length=1.5):

	z = np.array([0.0,0.0,1.0])
	cap_vecs -= cap_vecs[0]
	cap_com = np.average(cap_vecs, axis=0)
	rotM = M(cap_com, z)
	cap_vecs = dot(rotM, cap_vecs.T).T + bond_length*z

	name, form = filename.split('.')
	atoms = read(filename, format=form)
	G = atoms2graph(atoms)

	for e0,e1 in G.edges():

		ivec = G.nodes[e0]['ccoord']
		jvec = G.nodes[e1]['ccoord']

		dist = norm(ivec - jvec)

		if dist > 3.0:
			raise ValueError('bond length longer than 3.0 Å')

	all_cap_elems = []
	all_cap_coords = []
	for node,data in G.nodes(data=True):

		es = data['element_symbol']
		nbors = sorted([G.nodes[n]['element_symbol'] for n in G.neighbors(node)])
		nbor_vecs = np.array([G.nodes[n]['ccoord'] for n in G.neighbors(node)])

		if es == 'C' and (nbors == ['O','O'] or nbors == ['N','N'] or nbors == ['C','N']):

			concoords = np.r_[nbor_vecs, [data['ccoord']]]
			com = np.average(concoords, axis=0)
			concoords -= com
			rotM = M(z, concoords[-1])

			all_cap_elems.extend(cap_elems)
			rot_trans = dot(rotM, cap_vecs.T).T + com + concoords[-1]
			all_cap_coords.append(rot_trans)

	all_cap_coords = flatten(all_cap_coords)

	for es,vec in zip(all_cap_elems, all_cap_coords):
		atoms.append(Atom(es, vec))

	cappedG = atoms2graph(atoms)

	for e0,e1,data in cappedG.edges(data=True):

		ivec = cappedG.nodes[e0]['ccoord']
		jvec = cappedG.nodes[e1]['ccoord']
		ies = cappedG.nodes[e0]['element_symbol']
		jes = cappedG.nodes[e1]['element_symbol']
		inbors = list(cappedG.neighbors(e0))
		jnbors = list(cappedG.neighbors(e1))

		bond_elems = sorted([ies,jes])

		dist = np.round(norm(ivec - jvec),3)
		data['bond_length'] = str(dist)

		if bond_elems == ['C','O']:
			bt = 'A'
		elif bond_elems == ['C','N']:
			bt = 'A'
		elif bond_elems == ['N','N']:
			bt = 'A'
		elif bond_elems == ['C', 'C'] and len(inbors) == 4 and len(jnbors) == 4:
			bt = 'A'
		else:
			bt = 'S'

		data['bond_type'] = bt

	return cappedG, np.average(atoms.positions, axis=0)

def write_cif(G, com, filename):

	unit_cell = np.array([[50.0,0.0,0.0],[0.0,50.0,0.0],[0.0,0.0,50.0]])
	center = np.array([25.0,25.0,25.0])
	shift_vec = center - com

	for node,data in G.nodes(data=True):

		data['ccoord'] += shift_vec
		ccoord = data['ccoord']
		data['fcoord'] += np.dot(np.linalg.inv(unit_cell), ccoord)
		date = str(datetime.now()).split()[0]

	with open(filename, 'w') as out:
		out.write('data_sym_7_mc_4\n')
		out.write('_audit_creation_date              ' + date + '\n')
		out.write("_audit_creation_method            'tobacco_3.0'\n")
		out.write("_symmetry_space_group_name_H-M    'P1'\n")
		out.write('_symmetry_Int_Tables_number       1\n')
		out.write('_symmetry_cell_setting            triclinic\n')
		out.write('loop_\n')
		out.write('_symmetry_equiv_pos_as_xyz\n')
		out.write('  x,y,z\n')
		out.write('_cell_length_a                    50.0000\n')
		out.write('_cell_length_b                    50.0000\n')
		out.write('_cell_length_c                    50.0000\n')
		out.write('_cell_angle_alpha                 90.0000\n')
		out.write('_cell_angle_beta                  90.0000\n')
		out.write('_cell_angle_gamma                 90.0000\n')
		out.write('loop_\n')
		out.write('_atom_site_label\n')
		out.write('_atom_site_type_symbol\n')
		out.write('_atom_site_fract_x\n')
		out.write('_atom_site_fract_y\n')
		out.write('_atom_site_fract_z\n')
		out.write('_atom_site_U_iso_or_equiv\n')
		out.write('_atom_site_adp_type\n')
		out.write('_atom_site_occupancy\n')

		for node,data in G.nodes(data=True):

			vec = data['fcoord']
			line = [node, data['element_symbol'], vec[0], vec[1], vec[2], '0.00000', 'Uiso', '1.00']
			out.write('{:<5} {:<4} {:10.6f} {:10.6f} {:10.6f} {:<9} {:<6} {:<6}'.format(*line))
			out.write('\n')

		out.write('loop_\n')
		out.write('_geom_bond_atom_site_label_1\n')
		out.write('_geom_bond_atom_site_label_2\n')
		out.write('_geom_bond_distance\n')
		out.write('_geom_bond_site_symmetry_2\n')
		out.write('_ccdc_geom_bond_type\n')

		for e0,e1,data in G.edges(data=True):

			line = [e0, e1, data['bond_length'], data['bond_sym'], data['bond_type']]
			out.write('{:<5} {:<5} {:<7} {:<6} {:<3}'.format(*line))
			out.write('\n')

cap_elems = ['C', 'H', 'H', 'H']
cap_vecs = np.array([[-0.05628,  0.24065,  0.00000],
					 [-0.41294, -0.67971, -0.41306],
					 [-0.41295,  1.05855, -0.59053],
					 [-0.41295,  0.34311,  1.00359]])

cappedG, COM = cap_node('6c_Mg_1.xyz', cap_elems, cap_vecs)
write_cif(cappedG, COM, '6c_Mg_1.cif')




