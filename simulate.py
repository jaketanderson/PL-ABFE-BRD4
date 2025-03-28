import logging
import os
import pickle
import shutil
from typing import Iterable

import numpy as np
import ray
from openff.units import unit as openff_unit
from openmm import *
from openmm.app import *
from openmm.unit import *
from paprika import restraints
from paprika.restraints.openmm import apply_dat_restraint, apply_positional_restraints


@ray.remote(num_gpus=1, num_cpus=1)
def run_minimization(
    window: str,
    system: openmm.System,
    model: modeller.Modeller,
    restraints_to_apply: Iterable,
    sub_dir_name: str,
    simulation_parameters=None,
    data_dir_names={
        "initial_data": "initial_data",
        "prepared_data": "prepared_data",
        "working_data": "working_data",
    },
    suffix="",
):

    logger = logging.getLogger("PL_ABFE-BRD4")

    initial_data = data_dir_names["initial_data"]
    prepared_data = data_dir_names["prepared_data"]
    working_data = data_dir_names["working_data"]

    defaults = {
        "k_pos": 50 * kilocalorie / (mole * angstrom**2),
        "friction": 1 / picosecond,
        "timestep": 1 * femtoseconds,
        "tolerance": 0.001 * kilojoules_per_mole / nanometer,
        "maxIterations": 10_000,
    }

    unused_keys = set(simulation_parameters) - set(defaults)
    if unused_keys:
        logger.warning(
            f"The following simulation parameters are unused and will be ignored: {', '.join(unused_keys)}"
        )
    defaults.update(simulation_parameters)

    k_pos = defaults["k_pos"]
    friction = defaults["friction"]
    timestep = defaults["timestep"]
    tolerance = defaults["tolerance"]
    maxIterations = defaults["maxIterations"]

    directory = f"{working_data}/{sub_dir_name}/{window}"
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    if not "release_conf_restraints" in sub_dir_name:
        from_structure_path = f"{prepared_data}/aligned_dummy_structure{suffix}.pdb"
    else:
        from_structure_path = f"{prepared_data}/aligned_dummy_structure{suffix}_open_conf.pdb"

    shutil.copy(
        from_structure_path,
        f"{directory}/aligned_dummy_structure{suffix}.pdb",
    )
    
    
    with open(f"{prepared_data}/aligned_dummy_system{suffix}.xml", "r") as f:
        system = XmlSerializer.deserialize(f.read())

    restraints.openmm.apply_positional_restraints(
        f"{directory}/aligned_dummy_structure{suffix}.pdb",
        system,
        atom_name="DUM",
        force_group=15,
        k_pos=k_pos,
    )

    for restraint in restraints_to_apply:
        restraints.openmm.apply_dat_restraint(
            system,
            restraint,
            phase=restraints.utils.parse_window(window)[1],
            window_number=restraints.utils.parse_window(window)[0],
            force_group=10,
        )

    if not any([isinstance(force, CMMotionRemover) for force in system.getForces()]):
        system.addForce(CMMotionRemover())
        
    # to_remove = []
    # for i, force in enumerate(system.getForces()):
    #     if isinstance(force, CMMotionRemover):
    #         to_remove.append(i)
    # to_remove.sort(reverse=True)
    # for i in to_remove:
    #     system.removeForce(i)
        
    with open(f"{directory}/aligned_dummy_system{suffix}.xml", "w") as f:
        f.write(XmlSerializer.serialize(system))

    integrator = LangevinMiddleIntegrator(0 * kelvin, friction, timestep)
    integrator.setRandomNumberSeed(12345)
    platform = Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0"}
    simulation = Simulation(model.topology, system,
                            integrator, platform, properties)
    simulation.context.setPositions(model.positions)
    simulation.minimizeEnergy(tolerance=tolerance, maxIterations=maxIterations)

    positions = simulation.context.getState(getPositions=True).getPositions()

    with open(f"{directory}/minimized.pdb", "w") as f:
        PDBFile.writeFile(model.topology, positions, f)

    with open(f"{directory}/minimized_positions.pickle", "wb") as f:
        pickle.dump(positions, f)

    return


@ray.remote(num_gpus=1, num_cpus=1)
def run_heating_and_equil(
    window: str,
    system: openmm.System,
    model: modeller.Modeller,
    sub_dir_name: str,
    simulation_parameters=None,
    data_dir_names={
        "initial_data": "initial_data",
        "prepared_data": "prepared_data",
        "working_data": "working_data",
    },
    suffix="",
):

    logger = logging.getLogger("PL_ABFE-BRD4")

    initial_data = data_dir_names["initial_data"]
    prepared_data = data_dir_names["prepared_data"]
    working_data = data_dir_names["working_data"]

    defaults = {
        "temperatures": np.arange(0, 298.15, 10.0),
        "time_per_temp": 20 * picoseconds,
        "equilibration_time": 50 * picoseconds,  # This value will have the total time spent heating subtracted from it
        "friction": 1 / picosecond,
        "timestep": 1 * femtoseconds,
    }

    unused_keys = set(simulation_parameters) - set(defaults)
    if unused_keys:
        logger.warning(
            f"The following simulation parameters are unused and will be ignored: {', '.join(unused_keys)}"
        )
    defaults.update(simulation_parameters)

    temperatures = defaults["temperatures"]
    time_per_temp = defaults["time_per_temp"]
    equilibration_time = defaults["equilibration_time"]
    friction = defaults["friction"]
    timestep = defaults["timestep"]

    directory = f"{working_data}/{sub_dir_name}/{window}"

    with open(f"{directory}/aligned_dummy_system{suffix}.xml", "r") as f:
        system = XmlSerializer.deserialize(f.read())

    integrator = LangevinMiddleIntegrator(298.15 * kelvin, friction, timestep)
    integrator.setRandomNumberSeed(12345)
    platform = Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "mixed"}
    simulation = Simulation(model.topology, system,
                            integrator, platform, properties)
    with open(f"{directory}/minimized_positions.pickle", "rb") as f:
        minimized_positions = pickle.load(f)
    simulation.context.setPositions(minimized_positions)

    dcd_reporter = DCDReporter(f"{directory}/heating.dcd", 500)
    state_reporter = StateDataReporter(
        f"{directory}/heating.log",
        500,
        step=True,
        kineticEnergy=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        speed=True,
    )

    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(state_reporter)

    try:
        for temp in temperatures:
            integrator.setTemperature(temp * kelvin)
            simulation.step(int(time_per_temp / timestep))

        simulation.step(int((equilibration_time - (time_per_temp*len(temperatures))) / timestep))
        positions = simulation.context.getState(getPositions=True).getPositions()
    except Exception as e:
        with open("/home/jta002/workspace/PL-ABFE/PL-ABFE-BRD4/ERROR", "a+") as f:
            f.write(f"Heating exception {e} happened during window {window}!\n")

    with open(f"{directory}/heated.pdb", "w") as f:
        app.PDBFile.writeFile(model.topology, positions, f, keepIds=True)
    with open(f"{directory}/heated_positions.pickle", "wb") as f:
        pickle.dump(positions, f)

    with open(f"heating_info.debug", "a") as f:
        f.write(f"{window} has completed heating\n")

    return


@ray.remote(num_gpus=1, num_cpus=1)
def run_production(
    window: str,
    system: openmm.System,
    model: modeller.Modeller,
    sub_dir_name: str,
    simulation_parameters=None,
    data_dir_names={
        "initial_data": "initial_data",
        "prepared_data": "prepared_data",
        "working_data": "working_data",
    },
    suffix=""
):

    logger = logging.getLogger("PL_ABFE-BRD4")

    initial_data = data_dir_names["initial_data"]
    prepared_data = data_dir_names["prepared_data"]
    working_data = data_dir_names["working_data"]

    defaults = {
        "temperature": 298.15,
        "friction": 1 / picosecond,
        "timestep": 2 * femtoseconds,
        "production_time": 1 * nanoseconds,
        "dcd_reporter_frequency": 1000,
        "state_reporter_frequency": 1000,
    }

    unused_keys = set(simulation_parameters) - set(defaults)
    if unused_keys:
        logger.warning(
            f"The following simulation parameters are unused and will be ignored: {', '.join(unused_keys)}"
        )
    defaults.update(simulation_parameters)

    temperature = defaults["temperature"]
    friction = defaults["friction"]
    timestep = defaults["timestep"]
    production_time = defaults["production_time"]
    dcd_reporter_frequency = defaults["dcd_reporter_frequency"]
    state_reporter_frequency = defaults["state_reporter_frequency"]

    directory = f"{working_data}/{sub_dir_name}/{window}"

    # Load XML and input coordinates
    with open(f"{directory}/aligned_dummy_system{suffix}.xml", "r") as f:
        system = XmlSerializer.deserialize(f.read())

    integrator = LangevinMiddleIntegrator(temperature, friction, timestep)
    integrator.setRandomNumberSeed(12345)
    platform = Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "mixed"}
    simulation = Simulation(model.topology, system,
                            integrator, platform, properties)
    with open(f"{directory}/heated_positions.pickle", "rb") as f:
        heated_positions = pickle.load(f)
    simulation.context.setPositions(heated_positions)

    dcd_reporter = DCDReporter(
        f"{directory}/production.dcd", dcd_reporter_frequency)
    state_reporter = StateDataReporter(
        f"{directory}/production.log",
        state_reporter_frequency,
        step=True,
        kineticEnergy=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        speed=True,
    )

    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(state_reporter)

    try:
        simulation.step(int(production_time / timestep))
    except Exception as e:
        with open("/home/jta002/workspace/PL-ABFE/PL-ABFE-BRD4/ERROR", "a+") as f:
            f.write(f"Production exception {e} happened during window {window}!\n")
    positions = simulation.context.getState(getPositions=True).getPositions()

    with open(f"{directory}/production.pdb", "w") as f:
        app.PDBFile.writeFile(model.topology, positions, f, keepIds=True)

    return
