#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os
import pickle
import shutil
import sys
from copy import deepcopy
from importlib import reload

import numpy as np
import parmed as pmd
import ray
from openff.toolkit import ForceField as openffForceField
from openff.toolkit import Molecule as openffMolecule
from openff.toolkit import Topology as openffTopology
from openff.units import unit as openff_unit
from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from rdkit import Chem
from tqdm import tqdm

from openmm import *
from openmm.app import *
from openmm.unit import *

os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OE_LICENSE"] = os.environ["HOME"] + "oe_license.txt"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


ray.init(dashboard_port=8835, log_to_driver=True, logging_level="warning")

reload(logging)
logger = logging.getLogger("PL_ABFE-BRD4")
logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %I:%M:%S %p", level=logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

force_fields = "force_fields"
initial_data = "initial_data"
prepared_data = "prepared_data"
working_data = "working_data"
results_dirname = "results"

data_dir_names = {"initial_data": initial_data, "prepared_data": prepared_data,
                  "working_data": working_data, "results": results_dirname}


# ## Prepare the system

# In[2]:


def create_protein_mol():
    # Load ff14SB to assign charges to the protein
    ff14SB = ForceField("amber/ff14SB.xml")
    protein_pdb = PDBFile(f"{prepared_data}/4lys_protonated_vacuum.pdb")
    protein_system = ff14SB.createSystem(protein_pdb.topology)

    # Get ff14SB-assigned charges into a list
    protein_atomic_charges = []
    for force in protein_system.getForces():
        if isinstance(force, NonbondedForce):
            for i, atom in enumerate(protein_pdb.topology.atoms()):
                protein_atomic_charges.append(force.getParticleParameters(i)[0])

    # A little rdkit sanitization is necessary: we need to generate formal charges
    protein_rdkit_mol = Chem.MolFromMolFile(f"{prepared_data}/4lys_protonated_vacuum.sdf", sanitize=False)
    for atom in protein_rdkit_mol.GetAtoms():
        if atom.GetSymbol() == "N" and atom.GetExplicitValence() == 4:
            atom.SetFormalCharge(1)
    Chem.SanitizeMol(protein_rdkit_mol)
    Chem.MolToMolFile(protein_rdkit_mol, f"{prepared_data}/4lys_protonated_vacuum_clean.sdf")

    protein_mol = openffMolecule.from_file(f"{prepared_data}/4lys_protonated_vacuum_clean.sdf")

    # When translated to OpenFF Molecule, we have 13 extra hydrogens for some reason,
    # so here we create a new molecule that doesn't have these extra Hs.
    # The way we do this assumes all of the extraneous Hs are at the end of the atoms list
    protein_mol_new = openffMolecule()
    protein_mol_new.name = "4LYS"

    for i in range(protein_pdb.topology.getNumAtoms()):
        atom = protein_mol.atoms[i]
        metadata = {
            "residue_name": list(protein_pdb.topology.atoms())[i].residue.name,
            "residue_number": list(protein_pdb.topology.atoms())[i].residue.id,
        }
        protein_mol_new.add_atom(
            atom.atomic_number,
            atom.formal_charge,
            atom.is_aromatic,
            stereochemistry=atom.stereochemistry,
            name=list(protein_pdb.topology.atoms())[i].name,
            metadata=metadata,
        )

    # Add bonds to the new molecule
    for bond in protein_mol.bonds:
        if bond.atom1_index < protein_pdb.topology.getNumAtoms() and bond.atom2_index < protein_pdb.topology.getNumAtoms():
            protein_mol_new.add_bond(
                bond.atom1_index,
                bond.atom2_index,
                bond.bond_order,
                bond.is_aromatic,
                stereochemistry=bond.stereochemistry,
                fractional_bond_order=bond.fractional_bond_order,
            )

    # Add conformers
    for conformer in protein_mol.conformers:
        protein_mol_new.add_conformer(conformer[: protein_pdb.topology.getNumAtoms()])

    del protein_mol
    protein_mol = protein_mol_new
    assert protein_mol.n_atoms == protein_pdb.topology.getNumAtoms()
    assert protein_mol.n_bonds == protein_pdb.topology.getNumBonds()

    # Assign ff14SB charges to the OpenFF Molecule
    protein_mol.partial_charges = [ac.value_in_unit(elementary_charge)
                                   for ac in protein_atomic_charges] * openff_unit.elementary_charge
    # protein_mol.generate_unique_atom_names()

    sys.setrecursionlimit(10000)
    with open(f"{prepared_data}/protein_mol.pickle", "wb") as f:
        pickle.dump(protein_mol, f)

    return protein_mol


try:
    with open(f"{prepared_data}/protein_mol.pickle", "rb") as f:
        protein_mol = pickle.load(f)
except:
    protein_mol = create_protein_mol()


# In[3]:


def create_ligand_mol():
    ligand_mol = openffMolecule.from_file(f"{prepared_data}/4lys_D_2SJ_protonated_vacuum.mol2", file_format="mol2")
    ligand_mol.assign_partial_charges(partial_charge_method="am1bcc", use_conformers=ligand_mol.conformers)
    for atom in ligand_mol.atoms:
        atom.metadata["residue_name"] = "2SJ"

    sys.setrecursionlimit(10000)
    with open(f"{prepared_data}/ligand_mol.pickle", "wb") as f:
        pickle.dump(ligand_mol, f)

    return ligand_mol


try:
    with open(f"{prepared_data}/ligand_mol.pickle", "rb") as f:
        ligand_mol = pickle.load(f)
except:
    ligand_mol = create_ligand_mol()


# In[5]:


force_field = openffForceField(
    "openff-2.2.0.offxml",
    "ff14sb_off_impropers_0.0.4.offxml",
    "force_fields/GBSA-OBC2.offxml",
    load_plugins=True,
    allow_cosmetic_attributes=True,
)

openff_topology = openffTopology.from_molecules(molecules=[protein_mol, ligand_mol])
system = force_field.create_openmm_system(
    openff_topology,
    partial_bond_orders_from_molecules=[protein_mol, ligand_mol],
    charge_from_molecules=[protein_mol, ligand_mol],
    allow_nonintegral_charges=True,
)

openmm_positions = Quantity(
    value=[
        Vec3(
            pos[0].m_as(openff_unit.nanometer),
            pos[1].m_as(openff_unit.nanometer),
            pos[2].m_as(openff_unit.nanometer),
        )
        for pos in openff_topology.get_positions()
    ],
    unit=nanometer,
)
model = Modeller(openff_topology.to_openmm(ensure_unique_atom_names=False), openmm_positions)


# In[6]:


from paprika.build import align

structure = pmd.openmm.load_topology(
    model.topology,
    system,
    xyz=openmm_positions,
    condense_atom_types=False,
)

A1 = ":2SJ@C5"  # ":2SJ@C10x"
A2 = ":2SJ@C22"  # ":2SJ@C16x"

align.translate_to_origin(structure)
align.zalign(structure, A1, A2)


# In[7]:


P1 = ":67@CA"  # ":ILE@C365x"
P2 = ":88@CA"  # ":THR@C485x"
P3 = ":114@CA"  # ":PHE@C619x"

L1 = ":2SJ@O3"  # ":2SJ@O4x"
L2 = ":2SJ@C10"  # ":2SJ@C6x"
L3 = ":2SJ@C19"  # ":2SJ@C19x"


# In[8]:


# Ensure that P1 lies on the YZ plane
P1_original = structure[P1].coordinates[0]
P1_rotated = np.array([0, np.linalg.norm(P1_original[:2]), P1_original[2]])

theta_initial = np.arctan2(P1_original[1], P1_original[0])
theta_final = np.arctan2(P1_rotated[1], P1_rotated[0])

# Calculate the rotation angle in radians
delta_theta = theta_final - theta_initial
align.rotate_around_axis(structure, "z", delta_theta)

# Make sure that the new x coordinate of P1 is close to 0
print(f"The new x coordinate of P1 is: {structure[P1].coordinates[0][0]:0.3f}")

model = Modeller(model.topology, structure.positions.in_units_of(nanometer))

# N1_pos is same x and y as L1, but 5 Angstroms less in z
N1_pos = Quantity(
    value=Vec3(0, 0, (structure[L1].positions[0][2] - 5 * angstroms).value_in_unit(nanometer)),
    unit=nanometer,
)
# N2_pos is same x and z as N1_pos, but with a y value equal to to the y value of P1
N2_pos = Quantity(
    value=Vec3(
        0,
        structure[P1].positions[0][1].value_in_unit(nanometer),
        N1_pos[2].value_in_unit(nanometer),
    ),
    unit=nanometer,
)

N3_pos = Quantity(
    value=Vec3(
        0,
        N2_pos[1].value_in_unit(nanometer),
        (N2_pos[2] + (N2_pos[2] - structure[P1].positions[0][2])).value_in_unit(nanometer),
    ),
    unit=nanometer,
)


# In[9]:


# Give the dummy atoms mass of lead (207.2) and set the nonbonded forces for the dummy atoms to have charge 0 and LJ parameters epsilon=sigma=0.
for force in system.getForces():
    if isinstance(force, openmm.NonbondedForce):
        system.addParticle(mass=207.2)
        system.addParticle(mass=207.2)
        system.addParticle(mass=207.2)
        force.addParticle(0, 0, 0)
        force.addParticle(0, 0, 0)
        force.addParticle(0, 0, 0)

with open(f"{prepared_data}/aligned_dummy_system.xml", "w") as f:
    f.write(XmlSerializer.serialize(system))

all_positions = model.positions
all_positions.append(N1_pos)
all_positions.append(N2_pos)
all_positions.append(N3_pos)

chain = model.topology.addChain("5")
dummy_res1 = model.topology.addResidue(name="DM1", chain=chain)
dummy_res2 = model.topology.addResidue(name="DM2", chain=chain)
dummy_res3 = model.topology.addResidue(name="DM3", chain=chain)

model.topology.addAtom("DUM", Element.getBySymbol("Pb"), dummy_res1)
model.topology.addAtom("DUM", Element.getBySymbol("Pb"), dummy_res2)
model.topology.addAtom("DUM", Element.getBySymbol("Pb"), dummy_res3)

with open(f"{prepared_data}/aligned_dummy_structure.cif", "w+") as f:
    PDBxFile.writeFile(model.topology, all_positions, file=f, keepIds=True)
with open(f"{prepared_data}/aligned_dummy_structure.pdb", "w+") as f:
    PDBFile.writeFile(model.topology, all_positions, file=f, keepIds=True)

aligned_dummy_structure = pmd.openmm.load_topology(model.topology, system, xyz=all_positions, condense_atom_types=False)


# In[23]:


N1 = ":DM1@DUM"
N2 = ":DM2@DUM"
N3 = ":DM3@DUM"

ASP88_1 = ":45@N"  # Asp88 N
ASP88_2 = ":45@CA"  # Asp88 C_alpha
ASP88_3 = ":45@C"  # Asp88 C
ASP88_4 = ":46@N"  # Ala89 N


# In[24]:


for mask in [
    A1,
    A2,
    P1,
    P2,
    P3,
    L1,
    L2,
    L3,
    N1,
    N2,
    N3,
    ASP88_1,
    ASP88_2,
    ASP88_3,
    ASP88_4,
]:
    try:
        assert len(aligned_dummy_structure[mask].positions) == 1
    except:
        print(mask)


# In[11]:


deltaG_values = {}


# ## Calculate $\Delta G_{attach,p}$

# In[25]:


from paprika import restraints
from paprika.restraints.openmm import apply_dat_restraint, apply_positional_restraints

block_length = 5
block_separators = [0, 0.003, 0.010, 0.025, 0.050, 0.100, 0.200, 0.500, 1.000]
attach_p_fractions = []
i = 0
while i < len(block_separators):
    try:
        attach_p_fractions += np.linspace(block_separators[i], block_separators[i + 1], block_length).tolist()[:-1]
        i += 1
    except IndexError:
        attach_p_fractions += [1.0]
        break

attach_p_fractions = np.array(attach_p_fractions)

attach_p_restraints_static = []
attach_p_restraints_dynamic = []


P1_P2_distance = restraints.DAT_restraint()
P1_P2_distance.mask1 = P1
P1_P2_distance.mask2 = P2
P1_P2_distance.topology = aligned_dummy_structure
P1_P2_distance.auto_apr = False
P1_P2_distance.continuous_apr = False
P1_P2_distance.amber_index = False
P1_P2_vector = aligned_dummy_structure[P2].positions[0] - aligned_dummy_structure[P1].positions[0]
P1_P2_vector = [val.value_in_unit(angstrom) for val in P1_P2_vector]
P1_P2_distance.attach["target"] = openff_unit.Quantity(value=np.linalg.norm(P1_P2_vector), units=openff_unit.angstrom)
P1_P2_distance.attach["fraction_list"] = attach_p_fractions * 100
P1_P2_distance.attach["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
P1_P2_distance.attach["fc_initial"] = 0
P1_P2_distance.initialize()

attach_p_restraints_dynamic.append(P1_P2_distance)


P2_P3_distance = restraints.DAT_restraint()
P2_P3_distance.mask1 = P2
P2_P3_distance.mask2 = P3
P2_P3_distance.topology = aligned_dummy_structure
P2_P3_distance.auto_apr = False
P2_P3_distance.continuous_apr = False
P2_P3_distance.amber_index = False
P2_P3_vector = aligned_dummy_structure[P3].positions[0] - aligned_dummy_structure[P2].positions[0]
P2_P3_vector = [val.value_in_unit(angstrom) for val in P2_P3_vector]
P2_P3_distance.attach["target"] = openff_unit.Quantity(value=np.linalg.norm(P2_P3_vector), units=openff_unit.angstrom)
P2_P3_distance.attach["fraction_list"] = attach_p_fractions * 100
P2_P3_distance.attach["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
P2_P3_distance.initialize()

attach_p_restraints_dynamic.append(P2_P3_distance)


P1_P3_distance = restraints.DAT_restraint()
P1_P3_distance.mask1 = P1
P1_P3_distance.mask2 = P3
P1_P3_distance.topology = aligned_dummy_structure
P1_P3_distance.auto_apr = False
P1_P3_distance.continuous_apr = False
P1_P3_distance.amber_index = False
P1_P3_vector = aligned_dummy_structure[P3].positions[0] - aligned_dummy_structure[P1].positions[0]
P1_P3_vector = [val.value_in_unit(angstrom) for val in P1_P3_vector]
P1_P3_distance.attach["target"] = openff_unit.Quantity(value=np.linalg.norm(P1_P3_vector), units=openff_unit.angstrom)
P1_P3_distance.attach["fraction_list"] = attach_p_fractions * 100
P1_P3_distance.attach["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
P1_P3_distance.initialize()

attach_p_restraints_dynamic.append(P1_P3_distance)


Asp88_torsion = restraints.DAT_restraint()
Asp88_torsion.mask1 = ASP88_1
Asp88_torsion.mask2 = ASP88_2
Asp88_torsion.mask3 = ASP88_3
Asp88_torsion.mask4 = ASP88_4
Asp88_torsion.topology = aligned_dummy_structure
Asp88_torsion.auto_apr = False
Asp88_torsion.continuous_apr = False
Asp88_torsion.amber_index = False
Asp88_torsion.attach["target"] = 80  # Degrees
Asp88_torsion.attach["fraction_list"] = attach_p_fractions * 100
Asp88_torsion.attach["fc_final"] = 20  # kilocalorie/(mole*radian**2)
Asp88_torsion.initialize()

attach_p_restraints_dynamic.append(Asp88_torsion)


# Check restraints and create windows from dynamic restraints
restraints.restraints.check_restraints(attach_p_restraints_dynamic)
window_list = restraints.utils.create_window_list(attach_p_restraints_dynamic)


# In[13]:


# from simulate import run_heating_and_equil, run_minimization, run_production

# simulation_parameters = {
#     "k_pos": 50 * kilocalorie / (mole * angstrom**2),
#     "friction": 1 / picosecond,
#     "timestep": 1 * femtoseconds,
#     "tolerance": 0.0001 * kilojoules_per_mole / nanometer,
#     "maxIterations": 0,
# }

# futures = [
#     run_minimization.remote(
#         window,
#         system,
#         model,
#         attach_p_restraints_dynamic,
#         "attach_p_restraints",
#         simulation_parameters=simulation_parameters,
#         data_dir_names=data_dir_names,
#     )
#     for window in window_list
# ]
# _ = ray.get(futures)


# simulation_parameters = {
#     "temperatures": np.arange(0, 298.15, 10.0),
#     "time_per_temp": 20 * picoseconds,
#     "equilibration_time": 15 * nanoseconds,
#     "friction": 1 / picosecond,
#     "timestep": 1 * femtoseconds,
# }

# futures = [
#     run_heating_and_equil.remote(
#         window,
#         system,
#         model,
#         "attach_p_restraints",
#         simulation_parameters=simulation_parameters,
#         data_dir_names=data_dir_names,
#     )
#     for window in window_list
# ]
# _ = ray.get(futures)


# simulation_parameters = {
#     "temperature": 298.15,
#     "friction": 1 / picosecond,
#     "timestep": 2 * femtoseconds,
#     "production_time": 25 * nanoseconds,
#     "dcd_reporter_frequency": 50,
#     "state_reporter_frequency": 50,
# }

# futures = [
#     run_production.remote(
#         window,
#         system,
#         model,
#         "attach_p_restraints",
#         simulation_parameters=simulation_parameters,
#         data_dir_names=data_dir_names,
#     )
#     for window in window_list
# ]
# results = ray.get(futures)


# In[14]:


# import paprika.analysis as analysis

# free_energy = analysis.fe_calc()
# free_energy.topology = "heated.pdb"
# free_energy.trajectory = "production.dcd"
# free_energy.path = f"{working_data}/attach_p_restraints"
# free_energy.restraint_list = attach_p_restraints_static + attach_p_restraints_dynamic
# free_energy.collect_data(single_topology=False)
# free_energy.methods = ["mbar-autoc"]
# free_energy.boot_cycles = 10000
# free_energy.compute_free_energy(phases=["attach"])
# free_energy.save_results(f"{results_dirname}/deltaG_attach_p.json", overwrite=True)

# deltaG_attach_p = {
#     "fe": free_energy.results["attach"]["mbar-autoc"]["fe"],
#     "sem": free_energy.results["attach"]["mbar-autoc"]["sem"],
#     "fe_matrix": free_energy.results["attach"]["mbar-autoc"]["fe_matrix"],
#     "sem_matrix": free_energy.results["attach"]["mbar-autoc"]["sem_matrix"],
# }
# deltaG_values["deltaG_attach_p"] = deltaG_attach_p
# # print(deltaG_attach_p)
# # Vacuum {'fe': <Quantity(12.5876005, 'kilocalorie / mole')>, 'se': <Quantity(0.0282564208, 'kilocalorie / mole')>}
# # OBC2 {'fe': <Quantity(11.2408369, 'kilocalorie / mole')>, 'se': <Quantity(0.0685526213, 'kilocalorie / mole')>}
# # OBC_Logistic {'fe': <Quantity(11.9366452, 'kilocalorie / mole')>, 'se': <Quantity(0.0649317552, 'kilocalorie / mole')>}


# In[15]:


# list(np.diag(deltaG_attach_p["fe_matrix"].m_as(openff_unit.kilojoule / openff_unit.mole), k=1))


# ## Calculate $\Delta G_{attach,l}$

# In[26]:


from paprika import restraints
from paprika.restraints.openmm import apply_dat_restraint, apply_positional_restraints

block_length = 5
block_separators = [0, 0.003, 0.010, 0.025, 0.050, 0.100, 0.200, 0.500, 1.000]
attach_l_fractions = []
i = 0
while i < len(block_separators):
    try:
        attach_l_fractions += np.linspace(block_separators[i], block_separators[i + 1], block_length).tolist()[:-1]
        i += 1
    except IndexError:
        attach_l_fractions += [1.0]
        break

attach_l_fractions = np.array(attach_l_fractions)

# P1-P2-P3 distances, Asp88 ψ dihedral, D2, A3, A4, T4, T5, T6 (Fig. 1a)
P1_P2_distance = restraints.static_DAT_restraint(
    [P1, P2], [len(attach_l_fractions), 0, 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
P2_P3_distance = restraints.static_DAT_restraint(
    [P2, P3], [len(attach_l_fractions), 0, 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
P1_P3_distance = restraints.static_DAT_restraint(
    [P1, P2], [len(attach_l_fractions), 0, 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
Asp88_torsion = restraints.static_DAT_restraint(
    [ASP88_1, ASP88_2, ASP88_3, ASP88_4],
    [len(attach_l_fractions), 0, 0],
    "prepared_data/aligned_dummy_structure.pdb",
    20 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
P1_N2_distance = restraints.static_DAT_restraint(
    [P1, N2], [len(attach_l_fractions), 0, 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
P1_N2_N1_angle = restraints.static_DAT_restraint(
    [P1, N2, N1],
    [len(attach_l_fractions), 0, 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
P2_P1_N2_angle = restraints.static_DAT_restraint(
    [P2, P1, N2],
    [len(attach_l_fractions), 0, 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
P1_N2_N1_N3_torsion = restraints.static_DAT_restraint(
    [P1, N2, N1, N3],
    [len(attach_l_fractions), 0, 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
P2_P1_N2_N1_torsion = restraints.static_DAT_restraint(
    [P2, P1, N2, N1],
    [len(attach_l_fractions), 0, 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
P3_P2_P1_N2_torsion = restraints.static_DAT_restraint(
    [P2, P1, N2, N1],
    [len(attach_l_fractions), 0, 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)

attach_l_restraints_static = [
    P1_P2_distance,
    P2_P3_distance,
    P1_P3_distance,
    Asp88_torsion,
    P1_N2_distance,
    P1_N2_N1_angle,
    P2_P1_N2_angle,
    P1_N2_N1_N3_torsion,
    P2_P1_N2_N1_torsion,
    P3_P2_P1_N2_torsion,
]

# L1-L2-L3 distances, ligand dihedral when present (NOT PRESENT FOR THIS LIGAND), D1, A1, A2, T1, T2, T3 (Fig. 1a)
attach_l_restraints_dynamic = []

# L1-L2
L1_L2_distance = restraints.DAT_restraint()
L1_L2_distance.mask1 = L1
L1_L2_distance.mask2 = L2
L1_L2_distance.topology = aligned_dummy_structure
L1_L2_distance.auto_apr = False
L1_L2_distance.continuous_apr = False
L1_L2_distance.amber_index = False
L1_L2_vector = aligned_dummy_structure[L2].positions[0] - aligned_dummy_structure[L1].positions[0]
L1_L2_vector = [val.value_in_unit(angstrom) for val in L1_L2_vector]
L1_L2_distance.attach["target"] = openff_unit.Quantity(value=np.linalg.norm(L1_L2_vector), units=openff_unit.angstrom)
L1_L2_distance.attach["fraction_list"] = attach_l_fractions * 100
L1_L2_distance.attach["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
L1_L2_distance.attach["fc_initial"] = 0
L1_L2_distance.initialize()
attach_l_restraints_dynamic.append(L1_L2_distance)

# L2-L3
L2_L3_distance = restraints.DAT_restraint()
L2_L3_distance.mask1 = L2
L2_L3_distance.mask2 = L3
L2_L3_distance.topology = aligned_dummy_structure
L2_L3_distance.auto_apr = False
L2_L3_distance.continuous_apr = False
L2_L3_distance.amber_index = False
L2_L3_vector = aligned_dummy_structure[L3].positions[0] - aligned_dummy_structure[L2].positions[0]
L2_L3_vector = [val.value_in_unit(angstrom) for val in L2_L3_vector]
L2_L3_distance.attach["target"] = openff_unit.Quantity(value=np.linalg.norm(L2_L3_vector), units=openff_unit.angstrom)
L2_L3_distance.attach["fraction_list"] = attach_l_fractions * 100
L2_L3_distance.attach["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
L2_L3_distance.attach["fc_initial"] = 0
L2_L3_distance.initialize()
attach_l_restraints_dynamic.append(L2_L3_distance)

# L1-L3
L1_L3_distance = restraints.DAT_restraint()
L1_L3_distance.mask1 = L1
L1_L3_distance.mask2 = L3
L1_L3_distance.topology = aligned_dummy_structure
L1_L3_distance.auto_apr = False
L1_L3_distance.continuous_apr = False
L1_L3_distance.amber_index = False
L1_L3_vector = aligned_dummy_structure[L3].positions[0] - aligned_dummy_structure[L1].positions[0]
L1_L3_vector = [val.value_in_unit(angstrom) for val in L1_L3_vector]
L1_L3_distance.attach["target"] = openff_unit.Quantity(value=np.linalg.norm(L1_L3_vector), units=openff_unit.angstrom)
L1_L3_distance.attach["fraction_list"] = attach_l_fractions * 100
L1_L3_distance.attach["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
L1_L3_distance.attach["fc_initial"] = 0
L1_L3_distance.initialize()
attach_l_restraints_dynamic.append(L1_L3_distance)

# D1
N1_L1_distance = restraints.DAT_restraint()
N1_L1_distance.mask1 = N1
N1_L1_distance.mask2 = L1
N1_L1_distance.topology = aligned_dummy_structure
N1_L1_distance.auto_apr = False
N1_L1_distance.continuous_apr = False
N1_L1_distance.amber_index = False
N1_L1_vector = aligned_dummy_structure[L1].positions[0] - aligned_dummy_structure[N1].positions[0]
N1_L1_vector = [val.value_in_unit(angstrom) for val in N1_L1_vector]
N1_L1_distance.attach["target"] = openff_unit.Quantity(value=np.linalg.norm(N1_L1_vector), units=openff_unit.angstrom)
D1_BOUND_VALUE = openff_unit.Quantity(value=np.linalg.norm(
    N1_L1_vector), units=openff_unit.angstrom)  # This should be very close to 5A
N1_L1_distance.attach["fraction_list"] = attach_l_fractions * 100
N1_L1_distance.attach["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
N1_L1_distance.attach["fc_initial"] = 0
N1_L1_distance.initialize()
attach_l_restraints_dynamic.append(N1_L1_distance)

# A1
N2_N1_L1_angle = restraints.DAT_restraint()
N2_N1_L1_angle.mask1 = N2
N2_N1_L1_angle.mask2 = N1
N2_N1_L1_angle.mask3 = L1
N2_N1_L1_angle.topology = aligned_dummy_structure
N2_N1_L1_angle.auto_apr = False
N2_N1_L1_angle.continuous_apr = False
N2_N1_L1_angle.amber_index = False
N1_N2_vector = np.array(aligned_dummy_structure[N2].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(angstrom))
N1_L1_vector = np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(angstrom))
N2_N1_L1_angle.attach["target"] = np.degrees(
    np.arccos(np.dot(N1_N2_vector, N1_L1_vector) / (np.linalg.norm(N1_N2_vector) * np.linalg.norm(N1_L1_vector))))  # Degrees
N2_N1_L1_angle.attach["fraction_list"] = attach_l_fractions * 100
N2_N1_L1_angle.attach["fc_final"] = 100  # kilocalorie/(mole*radian**2)
N2_N1_L1_angle.initialize()
attach_l_restraints_dynamic.append(N2_N1_L1_angle)

# A2
N1_L1_L2_angle = restraints.DAT_restraint()
N1_L1_L2_angle.mask1 = N1
N1_L1_L2_angle.mask2 = L1
N1_L1_L2_angle.mask3 = L2
N1_L1_L2_angle.topology = aligned_dummy_structure
N1_L1_L2_angle.auto_apr = False
N1_L1_L2_angle.continuous_apr = False
N1_L1_L2_angle.amber_index = False
L1_N1_vector = np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(angstrom))
L1_L2_vector = np.array(aligned_dummy_structure[L2].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(angstrom))
N1_L1_L2_angle.attach["target"] = np.degrees(
    np.arccos(np.dot(L1_N1_vector, L1_L2_vector) / (np.linalg.norm(L1_N1_vector) * np.linalg.norm(L1_L2_vector))))  # Degrees
N1_L1_L2_angle.attach["fraction_list"] = attach_l_fractions * 100
N1_L1_L2_angle.attach["fc_final"] = 100  # kilocalorie/(mole*radian**2)
N1_L1_L2_angle.initialize()
attach_l_restraints_dynamic.append(N1_L1_L2_angle)

# T1
N3_N2_N1_L1_torsion = restraints.DAT_restraint()
N3_N2_N1_L1_torsion.mask1 = N3
N3_N2_N1_L1_torsion.mask2 = N2
N3_N2_N1_L1_torsion.mask3 = N1
N3_N2_N1_L1_torsion.mask4 = L1
N3_N2_N1_L1_torsion.topology = aligned_dummy_structure
N3_N2_N1_L1_torsion.auto_apr = False
N3_N2_N1_L1_torsion.continuous_apr = False
N3_N2_N1_L1_torsion.amber_index = False
N3_N2_vector = np.array(aligned_dummy_structure[N2].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N3].positions[0].value_in_unit(angstrom))
N2_N1_vector = np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N2].positions[0].value_in_unit(angstrom))
N1_L1_vector = np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(angstrom))
norm1, norm2 = np.cross(N3_N2_vector, N2_N1_vector), np.cross(N2_N1_vector, N1_L1_vector)
N3_N2_N1_L1_torsion.attach["target"] = np.degrees(np.arccos(np.dot(norm1, norm2) / (np.linalg.norm(norm1) * np.linalg.norm(norm2))))
N3_N2_N1_L1_torsion.attach["fraction_list"] = attach_l_fractions * 100
N3_N2_N1_L1_torsion.attach["fc_final"] = 100  # kilocalorie/(mole*radian**2)
N3_N2_N1_L1_torsion.initialize()
attach_l_restraints_dynamic.append(N3_N2_N1_L1_torsion)

# T2
N2_N1_L1_L2_torsion = restraints.DAT_restraint()
N2_N1_L1_L2_torsion.mask1 = N2
N2_N1_L1_L2_torsion.mask2 = N1
N2_N1_L1_L2_torsion.mask3 = L1
N2_N1_L1_L2_torsion.mask4 = L2
N2_N1_L1_L2_torsion.topology = aligned_dummy_structure
N2_N1_L1_L2_torsion.auto_apr = False
N2_N1_L1_L2_torsion.continuous_apr = False
N2_N1_L1_L2_torsion.amber_index = False
N2_N1_vector = np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N2].positions[0].value_in_unit(angstrom))
N1_L1_vector = np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(angstrom))
L1_L2_vector = np.array(aligned_dummy_structure[L2].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(angstrom))
norm1, norm2 = np.cross(N2_N1_vector, N1_L1_vector), np.cross(N1_L1_vector, L1_L2_vector)
N2_N1_L1_L2_torsion.attach["target"] = np.degrees(np.arccos(np.dot(norm1, norm2) / (np.linalg.norm(norm1) * np.linalg.norm(norm2))))
N2_N1_L1_L2_torsion.attach["fraction_list"] = attach_l_fractions * 100
N2_N1_L1_L2_torsion.attach["fc_final"] = 100  # kilocalorie/(mole*radian**2)
N2_N1_L1_L2_torsion.initialize()
attach_l_restraints_dynamic.append(N2_N1_L1_L2_torsion)

# T3
N1_L1_L2_L3_torsion = restraints.DAT_restraint()
N1_L1_L2_L3_torsion.mask1 = N1
N1_L1_L2_L3_torsion.mask2 = L1
N1_L1_L2_L3_torsion.mask3 = L2
N1_L1_L2_L3_torsion.mask4 = L3
N1_L1_L2_L3_torsion.topology = aligned_dummy_structure
N1_L1_L2_L3_torsion.auto_apr = False
N1_L1_L2_L3_torsion.continuous_apr = False
N1_L1_L2_L3_torsion.amber_index = False
N1_L1_vector = np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(angstrom))
L1_L2_vector = np.array(aligned_dummy_structure[L2].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(angstrom))
L2_L3_vector = np.array(aligned_dummy_structure[L3].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[L2].positions[0].value_in_unit(angstrom))
norm1, norm2 = np.cross(N1_L1_vector, L1_L2_vector), np.cross(L1_L2_vector, L2_L3_vector)
N1_L1_L2_L3_torsion.attach["target"] = np.degrees(np.arccos(np.dot(norm1, norm2) / (np.linalg.norm(norm1) * np.linalg.norm(norm2))))
N1_L1_L2_L3_torsion.attach["fraction_list"] = attach_l_fractions * 100
N1_L1_L2_L3_torsion.attach["fc_final"] = 100
N1_L1_L2_L3_torsion.initialize()
attach_l_restraints_dynamic.append(N1_L1_L2_L3_torsion)


# Check restraints and create windows from dynamic restraints
restraints.restraints.check_restraints(attach_l_restraints_static)
restraints.restraints.check_restraints(attach_l_restraints_dynamic)
window_list = restraints.utils.create_window_list(attach_l_restraints_dynamic)


# In[17]:


# from simulate import run_heating_and_equil, run_minimization, run_production

# simulation_parameters = {
#     "k_pos": 50 * kilocalorie / (mole * angstrom**2),
#     "friction": 1 / picosecond,
#     "timestep": 1 * femtoseconds,
#     "tolerance": 0.5 * kilojoules_per_mole / nanometer,
#     "maxIterations": 0,
# }

# futures = [
#     run_minimization.remote(
#         window,
#         system,
#         model,
#         attach_l_restraints_static + attach_l_restraints_dynamic,
#         "attach_l_restraints",
#         simulation_parameters=simulation_parameters,
#         data_dir_names=data_dir_names,
#     )
#     for window in window_list
# ]
# _ = ray.get(futures)


# simulation_parameters = {
#     "temperatures": np.arange(100, 298.15, 10.0),
#     "time_per_temp": 20 * picoseconds,
#     "equilibration_time": 15 * nanoseconds,
#     "friction": 1 / picosecond,
#     "timestep": 1 * femtoseconds,
# }

# futures = [
#     run_heating_and_equil.remote(
#         window,
#         system,
#         model,
#         "attach_l_restraints",
#         simulation_parameters=simulation_parameters,
#         data_dir_names=data_dir_names,
#     )
#     for window in window_list
# ]
# _ = ray.get(futures)


# simulation_parameters = {
#     "temperature": 298.15,
#     "friction": 1 / picosecond,
#     "timestep": 2 * femtoseconds,
#     "production_time": 25 * nanoseconds,
#     "dcd_reporter_frequency": 500,
#     "state_reporter_frequency": 500,
# }

# futures = [
#     run_production.remote(
#         window,
#         system,
#         model,
#         "attach_l_restraints",
#         simulation_parameters=simulation_parameters,
#         data_dir_names=data_dir_names,
#     )
#     for window in window_list
# ]
# _ = ray.get(futures)


# In[18]:


# free_energy = analysis.fe_calc()
# free_energy.topology = "heated.pdb"
# free_energy.trajectory = "production.dcd"
# free_energy.path = f"{working_data}/attach_l_restraints"
# free_energy.restraint_list = attach_l_restraints_static + attach_l_restraints_dynamic
# free_energy.collect_data(single_topology=False)
# free_energy.methods = ["mbar-autoc"]
# free_energy.boot_cycles = 10000
# free_energy.compute_free_energy(phases=["attach"])
# free_energy.save_results(f"{results_dirname}/deltaG_attach_l.json", overwrite=True)

# deltaG_attach_l = {
#     "fe": free_energy.results["attach"]["mbar-autoc"]["fe"],
#     "sem": free_energy.results["attach"]["mbar-autoc"]["sem"],
#     "fe_matrix": free_energy.results["attach"]["mbar-autoc"]["fe_matrix"],
#     "sem_matrix": free_energy.results["attach"]["mbar-autoc"]["sem_matrix"],
# }
# deltaG_values["deltaG_attach_l"] = deltaG_attach_l
# print(deltaG_attach_l)


# ## Calculate $\Delta G_{pull}$

# In[89]:


from paprika import restraints
from paprika.restraints.openmm import apply_dat_restraint, apply_positional_restraints

pull_distances = np.arange(D1_BOUND_VALUE.m_as(openff_unit.angstrom), 21 + 0.4, 0.4)

pull_restraints_static = []

# P1-P2-P3 distances, Asp88 ψ dihedral, D2, A3, A4, T4, T5, T6 (Fig. 1a)
P1_P2_distance = restraints.static_DAT_restraint(
    [P1, P2], [0, len(pull_distances), 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
P2_P3_distance = restraints.static_DAT_restraint(
    [P2, P3], [0, len(pull_distances), 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
P1_P3_distance = restraints.static_DAT_restraint(
    [P1, P2], [0, len(pull_distances), 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
Asp88_torsion = restraints.static_DAT_restraint(
    [ASP88_1, ASP88_2, ASP88_3, ASP88_4],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    20 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
P1_N2_distance = restraints.static_DAT_restraint(
    [P1, N2], [0, len(pull_distances), 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
P1_N2_N1_angle = restraints.static_DAT_restraint(
    [P1, N2, N1],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
P2_P1_N2_angle = restraints.static_DAT_restraint(
    [P2, P1, N2],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
P1_N2_N1_N3_torsion = restraints.static_DAT_restraint(
    [P1, N2, N1, N3],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
P2_P1_N2_N1_torsion = restraints.static_DAT_restraint(
    [P2, P1, N2, N1],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
P3_P2_P1_N2_torsion = restraints.static_DAT_restraint(
    [P2, P1, N2, N1],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)

pull_restraints_static += [
    P1_P2_distance,
    P2_P3_distance,
    P1_P3_distance,
    Asp88_torsion,
    P1_N2_distance,
    P1_N2_N1_angle,
    P2_P1_N2_angle,
    P1_N2_N1_N3_torsion,
    P2_P1_N2_N1_torsion,
    P3_P2_P1_N2_torsion,
]

# L1-L2-L3 distances, ligand dihedral when present (NOT PRESENT FOR THIS LIGAND), D1, A1, A2, T1, T2, T3 (Fig. 1a)
L1_L2_distance = restraints.static_DAT_restraint(
    [L1, L2], [0, len(pull_distances), 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
L2_L3_distance = restraints.static_DAT_restraint(
    [L2, L3], [0, len(pull_distances), 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
L1_L3_distance = restraints.static_DAT_restraint(
    [L1, L3], [0, len(pull_distances), 0], "prepared_data/aligned_dummy_structure.pdb", 5 *
    openff_unit.kilocalorie / (openff_unit.mole * openff_unit.angstrom**2)
)
N2_N1_L1_angle = restraints.static_DAT_restraint(
    [N2, N1, L1],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
N1_L1_L2_angle = restraints.static_DAT_restraint(
    [N1, L1, L2],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
N3_N2_N1_L1_torsion = restraints.static_DAT_restraint(
    [N3, N2, N1, L1],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
N2_N1_L1_L2_torsion = restraints.static_DAT_restraint(
    [N2, N1, L1, L2],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)
N1_L1_L2_L3_torsion = restraints.static_DAT_restraint(
    [N2, N1, L1, L2],
    [0, len(pull_distances), 0],
    "prepared_data/aligned_dummy_structure.pdb",
    100 * openff_unit.kilocalorie / (openff_unit.mole * openff_unit.radian**2),
)

pull_restraints_static += [
    L1_L2_distance,
    L2_L3_distance,
    L1_L3_distance,
    N2_N1_L1_angle,
    N1_L1_L2_angle,
    N3_N2_N1_L1_torsion,
    N2_N1_L1_L2_torsion,
    N1_L1_L2_L3_torsion,
]

pull_restraints_dynamic = []

# D1
N1_L1_distance = restraints.DAT_restraint()
N1_L1_distance.mask1 = N1
N1_L1_distance.mask2 = L1
N1_L1_distance.topology = aligned_dummy_structure
N1_L1_distance.auto_apr = False
N1_L1_distance.continuous_apr = False
N1_L1_distance.amber_index = False
N1_L1_distance.pull["target_initial"] = D1_BOUND_VALUE
N1_L1_distance.pull["target_final"] = 21
N1_L1_distance.pull["target_increment"] = 0.4
N1_L1_distance.pull["fc"] = 5  # kilocalorie/(mole*angstrom**2)
N1_L1_distance.initialize()
pull_restraints_dynamic.append(N1_L1_distance)

# Check restraints and create windows from dynamic restraints
restraints.restraints.check_restraints(pull_restraints_static)
restraints.restraints.check_restraints(pull_restraints_dynamic)
window_list = restraints.utils.create_window_list(pull_restraints_dynamic)


# In[ ]:


# from simulate import run_heating_and_equil, run_minimization, run_production

# simulation_parameters = {
#     "k_pos": 50 * kilocalorie / (mole * angstrom**2),
#     "friction": 1 / picosecond,
#     "timestep": 1 * femtoseconds,
#     "tolerance": 0.001 * kilojoules_per_mole / nanometer,
#     "maxIterations": 0,
# }

# futures = [
#     run_minimization.remote(
#         window,
#         system,
#         model,
#         pull_restraints_static + pull_restraints_dynamic,
#         "pull_restraints",
#         simulation_parameters=simulation_parameters,
#         data_dir_names=data_dir_names,
#     )
#     for window in window_list
# ]
# _ = ray.get(futures)


# simulation_parameters = {
#     "temperatures": np.arange(100, 298.15, 10.0),
#     "time_per_temp": 20 * picoseconds,
#     "equilibration_time": 40 * nanoseconds,
#     "friction": 1 / picosecond,
#     "timestep": 1 * femtoseconds,
# }

# futures = [
#     run_heating_and_equil.remote(
#         window,
#         system,
#         model,
#         "pull_restraints",
#         simulation_parameters=simulation_parameters,
#         data_dir_names=data_dir_names,
#     )
#     for window in window_list
# ]
# _ = ray.get(futures)


# simulation_parameters = {
#     "temperature": 298.15,
#     "friction": 1 / picosecond,
#     "timestep": 2 * femtoseconds,
#     "production_time": 100 * nanoseconds,
#     "dcd_reporter_frequency": 500,
#     "state_reporter_frequency": 500,
# }

# futures = [
#     run_production.remote(
#         window,
#         system,
#         model,
#         "pull_restraints",
#         simulation_parameters=simulation_parameters,
#         data_dir_names=data_dir_names,
#     )
#     for window in window_list
# ]
# _ = ray.get(futures)


# In[88]:


# free_energy = analysis.fe_calc()
# free_energy.topology = "heated.pdb"
# free_energy.trajectory = "production.dcd"
# free_energy.path = f"{working_data}/pull_restraints"
# free_energy.restraint_list = pull_restraints_static + pull_restraints_dynamic
# free_energy.collect_data(single_topology=False)
# free_energy.methods = ["mbar-autoc"]
# free_energy.boot_cycles = 10000
# free_energy.compute_free_energy(phases=["pull"])
# free_energy.save_results(f"{results_dirname}/deltaG_pull.json", overwrite=True)

# deltaG_pull = {
#     "fe": free_energy.results["pull"]["mbar-autoc"]["fe"],
#     "sem": free_energy.results["pull"]["mbar-autoc"]["sem"],
#     "fe_matrix": free_energy.results["pull"]["mbar-autoc"]["fe_matrix"],
#     "sem_matrix": free_energy.results["pull"]["mbar-autoc"]["sem_matrix"],
# }
# deltaG_values["deltaG_pull"] = deltaG_pull
# print(deltaG_pull)


# ## Calculate $\Delta G_{release,l}$

# In[17]:


# We only need to simulate the ligand

openff_topology = openffTopology.from_molecules(molecules=[ligand_mol])
system = force_field.create_openmm_system(
    openff_topology,
    partial_bond_orders_from_molecules=[ligand_mol],
    charge_from_molecules=[ligand_mol],
    allow_nonintegral_charges=True,
)

openmm_positions = Quantity(
    value=[
        Vec3(
            pos[0].m_as(openff_unit.nanometer),
            pos[1].m_as(openff_unit.nanometer),
            pos[2].m_as(openff_unit.nanometer),
        )
        for pos in openff_topology.get_positions()
    ],
    unit=nanometer,
)
model = Modeller(openff_topology.to_openmm(ensure_unique_atom_names=False), openmm_positions)

from paprika.build import align

structure = pmd.openmm.load_topology(
    model.topology,
    system,
    xyz=openmm_positions,
    condense_atom_types=False,
)

A1 = ":2SJ@C5"  # ":2SJ@C10x"
A2 = ":2SJ@C22"  # ":2SJ@C16x"

align.translate_to_origin(structure)
align.zalign(structure, A1, A2)

align.rotate_around_axis(structure, "z", delta_theta)


# In[18]:


# Give the dummy atoms mass of lead (207.2) and set the nonbonded forces for the dummy atoms to have charge 0 and LJ parameters epsilon=sigma=0.
for force in system.getForces():
    if isinstance(force, openmm.NonbondedForce):
        system.addParticle(mass=207.2)
        system.addParticle(mass=207.2)
        system.addParticle(mass=207.2)
        force.addParticle(0, 0, 0)
        force.addParticle(0, 0, 0)
        force.addParticle(0, 0, 0)

with open(f"{prepared_data}/aligned_dummy_system_ligand_only.xml", "w") as f:
    f.write(XmlSerializer.serialize(system))

all_positions = model.positions
all_positions.append(N1_pos)
all_positions.append(N2_pos)
all_positions.append(N3_pos)

chain = model.topology.addChain("5")
dummy_res1 = model.topology.addResidue(name="DM1", chain=chain)
dummy_res2 = model.topology.addResidue(name="DM2", chain=chain)
dummy_res3 = model.topology.addResidue(name="DM3", chain=chain)

model.topology.addAtom("DUM", Element.getBySymbol("Pb"), dummy_res1)
model.topology.addAtom("DUM", Element.getBySymbol("Pb"), dummy_res2)
model.topology.addAtom("DUM", Element.getBySymbol("Pb"), dummy_res3)

with open(f"{prepared_data}/aligned_dummy_structure_ligand_only.cif", "w+") as f:
    PDBxFile.writeFile(model.topology, all_positions, file=f, keepIds=True)
with open(f"{prepared_data}/aligned_dummy_structure_ligand_only.pdb", "w+") as f:
    PDBFile.writeFile(model.topology, all_positions, file=f, keepIds=True)

aligned_dummy_structure = pmd.openmm.load_topology(model.topology, system, xyz=all_positions, condense_atom_types=False)


# In[48]:


from paprika import restraints
from paprika.restraints.openmm import apply_dat_restraint, apply_positional_restraints

release_l_fractions = attach_l_fractions[::-1]

# L1-L2-L3 distances, ligand dihedral when present (NOT PRESENT FOR THIS LIGAND), D1, A1, A2, T1, T2, T3 (Fig. 1a)
release_l_restraints_dynamic = []

# L1-L2
L1_L2_distance = restraints.DAT_restraint()
L1_L2_distance.mask1 = L1
L1_L2_distance.mask2 = L2
L1_L2_distance.topology = aligned_dummy_structure
L1_L2_distance.auto_apr = False
L1_L2_distance.continuous_apr = False
L1_L2_distance.amber_index = False
L1_L2_vector = aligned_dummy_structure[L2].positions[0] - aligned_dummy_structure[L1].positions[0]
L1_L2_vector = [val.value_in_unit(angstrom) for val in L1_L2_vector]
L1_L2_distance.release["target"] = openff_unit.Quantity(value=np.linalg.norm(L1_L2_vector), units=openff_unit.angstrom)
L1_L2_distance.release["fraction_list"] = release_l_fractions * 100
L1_L2_distance.release["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
L1_L2_distance.initialize()
release_l_restraints_dynamic.append(L1_L2_distance)

# L2-L3
L2_L3_distance = restraints.DAT_restraint()
L2_L3_distance.mask1 = L2
L2_L3_distance.mask2 = L3
L2_L3_distance.topology = aligned_dummy_structure
L2_L3_distance.auto_apr = False
L2_L3_distance.continuous_apr = False
L2_L3_distance.amber_index = False
L2_L3_vector = aligned_dummy_structure[L3].positions[0] - aligned_dummy_structure[L2].positions[0]
L2_L3_vector = [val.value_in_unit(angstrom) for val in L2_L3_vector]
L2_L3_distance.release["target"] = openff_unit.Quantity(value=np.linalg.norm(L2_L3_vector), units=openff_unit.angstrom)
L2_L3_distance.release["fraction_list"] = release_l_fractions * 100
L2_L3_distance.release["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
L2_L3_distance.initialize()
release_l_restraints_dynamic.append(L2_L3_distance)

# L1-L3
L1_L3_distance = restraints.DAT_restraint()
L1_L3_distance.mask1 = L1
L1_L3_distance.mask2 = L3
L1_L3_distance.topology = aligned_dummy_structure
L1_L3_distance.auto_apr = False
L1_L3_distance.continuous_apr = False
L1_L3_distance.amber_index = False
L1_L3_vector = aligned_dummy_structure[L3].positions[0] - aligned_dummy_structure[L1].positions[0]
L1_L3_vector = [val.value_in_unit(angstrom) for val in L1_L3_vector]
L1_L3_distance.release["target"] = openff_unit.Quantity(value=np.linalg.norm(L1_L3_vector), units=openff_unit.angstrom)
L1_L3_distance.release["fraction_list"] = release_l_fractions * 100
L1_L3_distance.release["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
L1_L3_distance.initialize()
release_l_restraints_dynamic.append(L1_L3_distance)

# D1
N1_L1_distance = restraints.DAT_restraint()
N1_L1_distance.mask1 = N1
N1_L1_distance.mask2 = L1
N1_L1_distance.topology = aligned_dummy_structure
N1_L1_distance.auto_apr = False
N1_L1_distance.continuous_apr = False
N1_L1_distance.amber_index = False
N1_L1_vector = aligned_dummy_structure[L1].positions[0] - aligned_dummy_structure[N1].positions[0]
N1_L1_vector = [val.value_in_unit(angstrom) for val in N1_L1_vector]
N1_L1_distance.release["target"] = openff_unit.Quantity(value=np.linalg.norm(N1_L1_vector), units=openff_unit.angstrom)
D1_BOUND_VALUE = openff_unit.Quantity(value=np.linalg.norm(
    N1_L1_vector), units=openff_unit.angstrom)  # This should be very close to 5A
N1_L1_distance.release["fraction_list"] = release_l_fractions * 100
N1_L1_distance.release["fc_final"] = 5  # kilocalorie/(mole*angstrom**2)
N1_L1_distance.initialize()
release_l_restraints_dynamic.append(N1_L1_distance)

# A1
N2_N1_L1_angle = restraints.DAT_restraint()
N2_N1_L1_angle.mask1 = N2
N2_N1_L1_angle.mask2 = N1
N2_N1_L1_angle.mask3 = L1
N2_N1_L1_angle.topology = aligned_dummy_structure
N2_N1_L1_angle.auto_apr = False
N2_N1_L1_angle.continuous_apr = False
N2_N1_L1_angle.amber_index = False
N1_N2_vector = np.array(aligned_dummy_structure[N2].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(angstrom))
N1_L1_vector = np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(angstrom))
N2_N1_L1_angle.release["target"] = np.degrees(
    np.arccos(np.dot(N1_N2_vector, N1_L1_vector) / (np.linalg.norm(N1_N2_vector) * np.linalg.norm(N1_L1_vector))))  # Degrees
N2_N1_L1_angle.release["fraction_list"] = release_l_fractions * 100
N2_N1_L1_angle.release["fc_final"] = 100  # kilocalorie/(mole*radian**2)
N2_N1_L1_angle.initialize()
release_l_restraints_dynamic.append(N2_N1_L1_angle)

# A2
N1_L1_L2_angle = restraints.DAT_restraint()
N1_L1_L2_angle.mask1 = N1
N1_L1_L2_angle.mask2 = L1
N1_L1_L2_angle.mask3 = L2
N1_L1_L2_angle.topology = aligned_dummy_structure
N1_L1_L2_angle.auto_apr = False
N1_L1_L2_angle.continuous_apr = False
N1_L1_L2_angle.amber_index = False
L1_N1_vector = np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(angstrom))
L1_L2_vector = np.array(aligned_dummy_structure[L2].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(angstrom))
N1_L1_L2_angle.release["target"] = np.degrees(
    np.arccos(np.dot(L1_N1_vector, L1_L2_vector) / (np.linalg.norm(L1_N1_vector) * np.linalg.norm(L1_L2_vector))))  # Degrees
N1_L1_L2_angle.release["fraction_list"] = release_l_fractions * 100
N1_L1_L2_angle.release["fc_final"] = 100  # kilocalorie/(mole*radian**2)
N1_L1_L2_angle.initialize()
release_l_restraints_dynamic.append(N1_L1_L2_angle)

# T1
N3_N2_N1_L1_torsion = restraints.DAT_restraint()
N3_N2_N1_L1_torsion.mask1 = N3
N3_N2_N1_L1_torsion.mask2 = N2
N3_N2_N1_L1_torsion.mask3 = N1
N3_N2_N1_L1_torsion.mask4 = L1
N3_N2_N1_L1_torsion.topology = aligned_dummy_structure
N3_N2_N1_L1_torsion.auto_apr = False
N3_N2_N1_L1_torsion.continuous_apr = False
N3_N2_N1_L1_torsion.amber_index = False
N3_N2_vector = np.array(aligned_dummy_structure[N2].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N3].positions[0].value_in_unit(angstrom))
N2_N1_vector = np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N2].positions[0].value_in_unit(angstrom))
N1_L1_vector = np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(angstrom))
norm1, norm2 = np.cross(N3_N2_vector, N2_N1_vector), np.cross(N2_N1_vector, N1_L1_vector)
N3_N2_N1_L1_torsion.release["target"] = np.degrees(np.arccos(np.dot(norm1, norm2) / (np.linalg.norm(norm1) * np.linalg.norm(norm2))))
N3_N2_N1_L1_torsion.release["fraction_list"] = release_l_fractions * 100
N3_N2_N1_L1_torsion.release["fc_final"] = 100  # kilocalorie/(mole*radian**2)
N3_N2_N1_L1_torsion.initialize()
release_l_restraints_dynamic.append(N3_N2_N1_L1_torsion)

# T2
N2_N1_L1_L2_torsion = restraints.DAT_restraint()
N2_N1_L1_L2_torsion.mask1 = N2
N2_N1_L1_L2_torsion.mask2 = N1
N2_N1_L1_L2_torsion.mask3 = L1
N2_N1_L1_L2_torsion.mask4 = L2
N2_N1_L1_L2_torsion.topology = aligned_dummy_structure
N2_N1_L1_L2_torsion.auto_apr = False
N2_N1_L1_L2_torsion.continuous_apr = False
N2_N1_L1_L2_torsion.amber_index = False
N2_N1_vector = np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N2].positions[0].value_in_unit(angstrom))
N1_L1_vector = np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(angstrom))
L1_L2_vector = np.array(aligned_dummy_structure[L2].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(angstrom))
norm1, norm2 = np.cross(N2_N1_vector, N1_L1_vector), np.cross(N1_L1_vector, L1_L2_vector)
N2_N1_L1_L2_torsion.release["target"] = np.degrees(np.arccos(np.dot(norm1, norm2) / (np.linalg.norm(norm1) * np.linalg.norm(norm2))))
N2_N1_L1_L2_torsion.release["fraction_list"] = release_l_fractions * 100
N2_N1_L1_L2_torsion.release["fc_final"] = 100  # kilocalorie/(mole*radian**2)
N2_N1_L1_L2_torsion.initialize()
release_l_restraints_dynamic.append(N2_N1_L1_L2_torsion)

# T3
N1_L1_L2_L3_torsion = restraints.DAT_restraint()
N1_L1_L2_L3_torsion.mask1 = N1
N1_L1_L2_L3_torsion.mask2 = L1
N1_L1_L2_L3_torsion.mask3 = L2
N1_L1_L2_L3_torsion.mask4 = L3
N1_L1_L2_L3_torsion.topology = aligned_dummy_structure
N1_L1_L2_L3_torsion.auto_apr = False
N1_L1_L2_L3_torsion.continuous_apr = False
N1_L1_L2_L3_torsion.amber_index = False
N1_L1_vector = np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[N1].positions[0].value_in_unit(angstrom))
L1_L2_vector = np.array(aligned_dummy_structure[L2].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[L1].positions[0].value_in_unit(angstrom))
L2_L3_vector = np.array(aligned_dummy_structure[L3].positions[0].value_in_unit(
    angstrom)) - np.array(aligned_dummy_structure[L2].positions[0].value_in_unit(angstrom))
norm1, norm2 = np.cross(N1_L1_vector, L1_L2_vector), np.cross(L1_L2_vector, L2_L3_vector)
N1_L1_L2_L3_torsion.release["target"] = np.degrees(np.arccos(np.dot(norm1, norm2) / (np.linalg.norm(norm1) * np.linalg.norm(norm2))))
N1_L1_L2_L3_torsion.release["fraction_list"] = attach_l_fractions * 100
N1_L1_L2_L3_torsion.release["fc_final"] = 100
N1_L1_L2_L3_torsion.initialize()
release_l_restraints_dynamic.append(N1_L1_L2_L3_torsion)


# Check restraints and create windows from dynamic restraints
restraints.restraints.check_restraints(release_l_restraints_dynamic)
window_list = restraints.utils.create_window_list(release_l_restraints_dynamic)


# In[50]:


from simulate import run_heating_and_equil, run_minimization, run_production

simulation_parameters = {
    "k_pos": 50 * kilocalorie / (mole * angstrom**2),
    "friction": 1 / picosecond,
    "timestep": 1 * femtoseconds,
    "tolerance": 0.0001 * kilojoules_per_mole / nanometer,
    "maxIterations": 0,
}

futures = [
    run_minimization.remote(
        window,
        system,
        model,
        release_l_restraints_dynamic,
        "release_l_restraints",
        simulation_parameters=simulation_parameters,
        data_dir_names=data_dir_names,
        suffix="_ligand_only",
    )
    for window in window_list
]
_ = ray.get(futures)


simulation_parameters = {
    "temperatures": np.arange(100, 298.15, 10.0),
    "time_per_temp": 50 * picoseconds,
    "equilibration_time": 15 * nanoseconds,
    "friction": 1 / picosecond,
    "timestep": 1 * femtoseconds,
}

futures = [
    run_heating_and_equil.remote(
        window,
        system,
        model,
        "release_l_restraints",
        simulation_parameters=simulation_parameters,
        data_dir_names=data_dir_names,
        suffix="_ligand_only",
    )
    for window in window_list
]
_ = ray.get(futures)


simulation_parameters = {
    "temperature": 298.15,
    "friction": 1 / picosecond,
    "timestep": 2 * femtoseconds,
    "production_time": 25 * nanoseconds,
    "dcd_reporter_frequency": 500,
    "state_reporter_frequency": 500,
}

futures = [
    run_production.remote(
        window,
        system,
        model,
        "release_l_restraints",
        simulation_parameters=simulation_parameters,
        data_dir_names=data_dir_names,
        suffix="_ligand_only",
    )
    for window in window_list
]
_ = ray.get(futures)


# In[ ]:


free_energy = analysis.fe_calc()
free_energy.topology = "heated.pdb"
free_energy.trajectory = "production.dcd"
free_energy.path = f"{working_data}/release_l_restraints"
free_energy.restraint_list = release_l_restraints_dynamic
free_energy.collect_data(single_topology=False)
free_energy.methods = ["mbar-autoc"]
free_energy.boot_cycles = 10000
free_energy.compute_free_energy(phases=["release"])
free_energy.save_results(f"{results_dirname}/deltaG_release_l.json", overwrite=True)

deltaG_release_l = {
    "fe": free_energy.results["pull"]["mbar-autoc"]["fe"],
    "sem": free_energy.results["pull"]["mbar-autoc"]["sem"],
    "fe_matrix": free_energy.results["pull"]["mbar-autoc"]["fe_matrix"],
    "sem_matrix": free_energy.results["pull"]["mbar-autoc"]["sem_matrix"],
}
deltaG_values["deltaG_release_l"] = deltaG_release_l
print(deltaG_release_l)


# In[ ]:



