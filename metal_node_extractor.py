import re
import numpy as np
import networkx as nx
import glob
from pymatgen.core import periodic_table
from itertools import chain
from ase.io import read, write
from ase import neighborlist
from numpy import dot
from numpy.linalg import norm

def PBC3DF_sym(vec1, vec2):
	"""
		applies periodic boundary to distance between vec1 and vec2 (fractional coordinates)
	"""
	dist = vec1 - vec2
	sym_dist = [(1.0, dim - 1.0) if dim > 0.5 else (-1.0, dim + 1.0) if dim < -0.5 else (0, dim) for dim in dist]
	sym = np.array([s[0] for s in sym_dist])
	ndist = np.array([s[1] for s in sym_dist])

	return ndist, sym

def nn(string):
	return re.sub('[^a-zA-Z]','', string)

def nl(string):
	return re.sub('[^0-9]','', string)

def neighborhood(G, n, cutoff=2):

	nbors = nx.single_source_shortest_path_length(G, n, cutoff=cutoff)

	return nbors

def flatten(L):

	return list(chain(*L))

def read_cif(cif):

	atoms = read(cif, format='cif')
	atoms.set_pbc(True)
	cutoffs = neighborlist.natural_cutoffs(atoms)
	unit_cell = atoms.get_cell()

	neighborlist.primitive_neighbor_list
	NL = neighborlist.NewPrimitiveNeighborList(cutoffs, use_scaled_positions=True, self_interaction=False)
	NL.build([True, True, True], unit_cell, atoms.get_scaled_positions())

	G = nx.Graph()

	for a,fcoord in zip(atoms, atoms.get_scaled_positions()):
		G.add_node(a.symbol + str(a.index), element_symbol=a.symbol, fcoord=fcoord)

	for a in G.nodes():
		
		nbors = [atoms[i].symbol + str(atoms[i].index) for i in NL.get_neighbors(int(nl(a)))[0]]
		
		for nbor in nbors:
			G.add_edge(a, nbor)

	return G, np.asarray(unit_cell).T

def find_clusters(G, unit_cell, coordination_shells=5, niter=5, mode='single_node', outside_cycle_elements=('H','O')):

	metals = [n for n in G if periodic_table.Element(G.nodes[n]['element_symbol']).is_metal]
	unique_shells = []
	clusters = []

	outside_cycle_elements = set(list(outside_cycle_elements) + [G.nodes[n]['element_symbol'] for n in metals])

	for m in metals:

		start = m
		
		start_shell = list(neighborhood(G, start, cutoff=coordination_shells))
		full_shell = list(set([n for n in start_shell if periodic_table.Element(G.nodes[n]['element_symbol']).is_metal]))
	
		for i in range(niter):

			metals = [n for n in full_shell if periodic_table.Element(G.nodes[n]['element_symbol']).is_metal]
			full_shell = set(flatten([list(neighborhood(G, n, cutoff=coordination_shells)) for n in metals]))

		first_shell = flatten([list(G.neighbors(m)) for m in metals])
		full_shell = sorted(list(full_shell) + first_shell)
		
		if full_shell not in unique_shells:

			unique_shells.append(full_shell)
			SG = G.subgraph(full_shell)
			cycles = flatten(list(nx.cycle_basis(SG)))
			cluster = SG.subgraph([n for n in SG.nodes() if (n in cycles or n in first_shell or SG.nodes[n]['element_symbol'] in outside_cycle_elements)])

			if len(cluster.nodes()) == 1:

				single_atom = list(cluster.nodes())[0]
				cluster = SG.subgraph(list(SG.neighbors(single_atom)) + [single_atom])

			fcoords = [data['fcoord'] for n,data in cluster.nodes(data=True)]
			anchor = fcoords[0]
	
			for node in cluster:
			
				fcoord = cluster.nodes[node]['fcoord']
				dist, sym = PBC3DF_sym(anchor, fcoord)
				cluster.nodes[node]['ccoord'] = dot(unit_cell, fcoord + sym)

			clusters.append(cluster)

			if mode == 'single_node':
				break

		else:
			continue

	return clusters

def write_clusters(clusters, fname=''):

	for i, cluster in enumerate(clusters):

		with open(fname + '_node' + str(i) + '.xyz', 'w') as out:
	
			out.write(str(len(cluster.nodes())) + '\n')
			out.write('No. 3 The Larch\n')
	
			for node, data in cluster.nodes(data=True):
	
				symbol = data['element_symbol']
				ccoord = data['ccoord']
				line = [symbol, ccoord[0], ccoord[1], ccoord[2]]
				out.write('{:<4} {:<10.5f} {:<10.5f} {:<10.5f}'.format(*line))
				out.write('\n')

def extract_clusters(cifs, coordination_shells=3, niter=5, mode='single_node', outside_cycle_elements=('H','O')):

	for cif in cifs:

		G, unit_cell = read_cif(cif)
		name = cif.split('/')[-1].split('.')[0]
		cluster = find_clusters(G, unit_cell, coordination_shells=coordination_shells, 
								niter=niter, mode=mode, outside_cycle_elements=outside_cycle_elements)
		write_clusters(cluster, fname=name)

#cifs = glob.glob('cifs1/*.cif')
#extract_clusters(cifs, mode='none')


