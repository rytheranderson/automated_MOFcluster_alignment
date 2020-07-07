import re
import networkx as nx
import numpy as np
from numpy import cross, eye, dot
from numpy.linalg import norm
from scipy.linalg import expm
from ase.io import read, write
from ase import neighborlist, Atom, Atoms
from itertools import chain, combinations
from superimposition import SVDSuperimposer

metals = ['Al', 'Cd', 'Ce', 'Co', 'Cr', 'Cu', 'Eu', 'Fe', 'In', 'Ni', 'Mg', 'Mn', 'Tb', 'Ti', 'V', 'Zn', 'Zr']

def flatten(L):
	return list(chain(*L))

def nl(string):
	return re.sub('[^0-9]','', string)

def nn(string):
	return re.sub('[0-9]','', string)

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
		cos_ang = dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
		ang = np.arccos(cos_ang)
		return R(ax_norm, ang)
	else:
		return eye(3)

def superimpose(a0, a1, max_permute=6):
	
	S = SVDSuperimposer()

	a0 = np.asarray(a0)
	a1 = np.asarray(a1)

	S.set(a1,a0)
	S.run()
	rmsd = S.get_rms()
	rot,trans = S.get_rotran()

	return rmsd, rot, trans

def atoms2graph(atoms, kwargs={}):

	unit_cell = atoms.get_cell()
	cutoffs = neighborlist.natural_cutoffs(atoms)
	NL = neighborlist.NewPrimitiveNeighborList(cutoffs, self_interaction=False, skin=0.1) # default atom cutoffs work well
	NL.build([False, False, False], unit_cell, atoms.positions)

	G = nx.Graph()

	for a, ccoord in zip(atoms, atoms.positions):
		ind = a.index + 1
		G.add_node(a.symbol + str(ind), element_symbol=a.symbol, ccoord=ccoord, fcoord=np.array([0.0,0.0,0.0]), freeze_code='0', **kwargs)

	for a in G.nodes():
		
		nbors = [atoms[i].symbol + str(atoms[i].index+1) for i in NL.get_neighbors(int(nl(a))-1)[0]]
		
		for nbor in nbors:
			G.add_edge(a, nbor, bond_type='', bond_sym='.', bond_length=0)

	return G

def find_carboxylates(G):

	all_carboxylates = []

	for i,data in G.nodes(data=True):

		isym = data['element_symbol']
		nbors = list(G.neighbors(i))
		nbor_symbols = [G.nodes[n]['element_symbol'] for n in nbors]
		nbor_coords = [G.nodes[n]['ccoord'] for n in nbors]

		if isym == 'C' and len([f for f in [nsym == 'O' for nsym in nbor_symbols] if f]) == 2:

			O_coords = [vec for vec,oflag in zip(nbor_coords, [nsym == 'O' for nsym in nbor_symbols]) if oflag]
			carboxylate_nodes = [i] + nbors
			carboxylate_coords = [data['ccoord']] + O_coords
			all_carboxylates.append((carboxylate_nodes, np.asarray(carboxylate_coords)))

	return all_carboxylates

def write_gjf(G, filename, sm=1, basis_org='6-31*', basis_metal='LANL2DZ', commands='Opt', metal_pseudo=(True, 'LANL2DZ')):

	elems = [data['element_symbol'] for n,data in G.nodes(data=True)]
	present_metals = set([e for e in elems if e in metals])
	present_non_metals = set([e for e in elems if e not in metals])

	if basis_org != basis_metal:
		line = '#P B3LYP/GEN'
	else:
		line = '#P B3LYP/' + basis_org
	if metal_pseudo[0]:
		line += ' Pseudo=Read'
	line += ' '
	line += commands

	with open(filename, 'w') as gjf:

		gjf.write('%chk=' + filename.split('.')[0] + '.chk\n')
		gjf.write('%nprocshared=16\n')
		gjf.write('%mem=4GB\n')
		gjf.write(line + '\n')
		gjf.write('\n')

		gjf.write('metal linker addition calculation\n')
		gjf.write('\n')
		gjf.write('0 ' + str(sm) + '\n')

		for n,data in G.nodes(data=True):

			sym = data['element_symbol']
			fc = data['freeze_code']
			ccoord = data['ccoord']
			vec = [np.round(v,4) for v in ccoord]

			gjf.write('{:<5} {:<5} {:<12} {:<12} {:<12}'.format(sym, fc, vec[0], vec[1], vec[2]))
			gjf.write('\n')

		gjf.write('\n')

		if basis_org != basis_metal:
			
			gjf.write(' '.join(present_metals) + ' 0\n')
			gjf.write(basis_metal + '\n')
			gjf.write('****\n')
	
			gjf.write(' '.join(present_non_metals) + ' 0\n')
			gjf.write(basis_org + '\n')
			gjf.write('****\n')
			gjf.write('\n')
	
		if metal_pseudo[0]:
	
			gjf.write(' '.join(present_metals) + ' 0\n')
			gjf.write(metal_pseudo[1] + '\n')
			gjf.write('\n')

def add_linkers(node_filename, linker_filename, N_linkers, mode='add', charge_balance_with_linkers=True, write_format='xyz', gjf_kwargs={}):

	if mode not in ('replace', 'add'):
		raise ValueError('mode argument is not recognized, it should be "replace" or "add"')

	Nname, Nform = node_filename.split('.')
	Natoms = read(node_filename, format=Nform)
	Natoms.positions -= np.average(Natoms.positions, axis=0)
	NG = atoms2graph(Natoms, kwargs={'flag':'node'})

	# freeze metals and non-carboxylate oxygens
	for node,data in NG.nodes(data=True):

		nbor_symbols = [NG.nodes[n]['element_symbol'] for n in NG.neighbors(node)]

		if data['element_symbol'] in metals:
			data['freeze_code'] = '-1'
		elif data['element_symbol'] == 'O' and 'C' not in nbor_symbols:
			data['freeze_code'] = '-1'

	Lname, Lform = linker_filename.split('.')
	Latoms = read(linker_filename, format=Lform)
	Latoms.positions -= np.average(Latoms.positions, axis=0)
	LG = atoms2graph(Latoms, kwargs={'flag':'linker'})

	L_carboxylate = find_carboxylates(LG)
	
	if len(L_carboxylate) > 1:
		raise ValueError('more than one carboxylate detected in', linker_filename)
	else:
		L_carboxylate = L_carboxylate[0]
		L_carboxylate_coords = L_carboxylate[1]

	N_carboxylates = find_carboxylates(NG)

	if len(N_carboxylates) < N_linkers:
		raise ValueError('fewer node carboxylates were detected than N_linkers')

	# consider all combinations of linker sites
	count = 0
	for comb in combinations(N_carboxylates, N_linkers):

		G = nx.Graph()
		remove_nodes = []
		max_ind = 0

		# add all node atoms to the combined graph
		for N_node, data in NG.nodes(data=True):
			
			ind = int(nl(N_node))
			G.add_node(N_node, **data)

			if ind > max_ind:
				max_ind = ind

		# find rotation and translation for linker coordinates and add the linkers
		for N_nodes, N_carboxylate_coords in comb:

			# remove only node carboxylates + caps where a linker is being placed if in replace mode
			if mode == 'replace':
				remove_nodes.extend(N_nodes)

			rmsd,rot,trans = superimpose(L_carboxylate_coords, N_carboxylate_coords)

			if rmsd > 0.1:
				raise ValueError('RMSD', np.round(rmsd,5), 'is higher than expected for the alignment of node carboxylate with linker carboxylate')

			for L_node, data in LG.copy().nodes(data=True):
				max_ind += 1
				data['ccoord'] = dot(data['ccoord'], rot) + trans
				G.add_node(nn(L_node) + str(max_ind), **data)

		# remove all node carboxylates + caps if in add mode
		if mode == 'add':
			for N_nodes, N_carboxylate_coords in N_carboxylates:
				remove_nodes.extend(N_nodes)

		# remove the relavent nodes depending on the mode (see above)
		for node in remove_nodes:
			G.remove_node(node)

		# add the number of linkers as "slots" remaining far from the metals to charge/atom balance
		if charge_balance_with_linkers and mode == 'add':
			
			other_N_carboxylates = [l for l in N_carboxylates if l not in comb]
			for oN_nodes, oN_carboxylate_coords in other_N_carboxylates:
				
				axis = oN_carboxylate_coords[0]/norm(oN_carboxylate_coords[0])
				rmsd,rot,trans = superimpose(L_carboxylate_coords, oN_carboxylate_coords)
		
				for L_node, data in LG.copy().nodes(data=True):
					max_ind += 1
					data['ccoord'] = dot(data['ccoord'], rot) + trans + 15.0 * axis
					G.add_node(nn(L_node) + str(max_ind), **data)

		# write unique outputs
		atoms = Atoms()
		for nodes, data in G.nodes(data=True):
			atoms.append(Atom(data['element_symbol'], data['ccoord']))

		fname = '_'.join([node_filename.split('.')[0], linker_filename.split('.')[0], str(N_linkers), str(count)]) + '.' + write_format
		
		if write_format == 'gjf':
			write_gjf(G, fname, **gjf_kwargs)
		elif write_format == 'xyz':
			write(fname, atoms, format='xyz')

		count += 1

add_linkers('Cu_paddlewheel.xyz', 'BDC_capped.xyz', 4, write_format='gjf', mode='add', gjf_kwargs={'sm':3, 'commands':'Opt SCF=(XQC, MaxConventionalCycles=100)'})
#write_cif(cappedG, COM, '6c_Mg_1.cif')




