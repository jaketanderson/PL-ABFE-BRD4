import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md
from paprika import utils
from tqdm import tqdm
import math
import os
from openff.units import unit as openff_unit
from matplotlib.backends.backend_pdf import PdfPages


def generate_histograms(restraints, phase_name, window_list, structure, data_dir_names):

    working_data = data_dir_names["working_data"]
    results = data_dir_names["results"]
    pdf = PdfPages(f"{results}/{phase_name}_OBC2.pdf")
    for counter, restraint in enumerate(restraints):
        shift_to_avoid_split = False
        minimum, maximum = (1e9, -1e9)
        fig = plt.figure()
        data_list, label_list = [], []

        target = None
        phase = phase_name.split("_")[0]
        match  phase:
            case "attach":
                target = restraint.attach["target"].m
                initial_fc = restraint.attach["fc_initial"]
                if initial_fc is None:
                    initial_fc = 0
                else:
                    initial_fc = initial_fc.m
                overall_force_constant_diff = initial_fc - restraint.attach["fc_final"].m
            case "pull":
                # target = restraint.pull["target"].m
                target = 0
                # initial_fc = restraint.pull["fc_initial"]
                # if initial_fc is None:
                #     initial_fc = 0
                # else:
                #     intiial_fc = initial_fc.m
                # overall_force_constant_diff = initial_fc - restraint.pull["fc_final"].m
                overall_force_constant_diff = 1e5
            case "release":
                target = restraint.release["target"].m
                initial_fc = restraint.release["fc_initial"]
                if initial_fc is None:
                    initial_fc = 0
                else:
                    initial_fc = initial_fc.m
                overall_force_constant_diff = initial_fc - restraint.release["fc_final"].m
            
        static_rest = abs(overall_force_constant_diff) < 1e-3
        
        for window in tqdm(window_list):
            top = f"{working_data}/{phase_name}/{window}/minimized.pdb"
            traj = md.load(f"{working_data}/{phase_name}/{window}/heating.dcd", top=top) + md.load(f"{working_data}/{phase_name}/{window}/production.dcd", top=top)
            mask_list = [a for a in [restraint.mask1, restraint.mask2, restraint.mask3, restraint.mask4] if a is not None]
            index_list = [utils.index_from_mask(structure, mask)[0] for mask in mask_list]
            decoration = None

            # Distance
            if len(index_list) == 2:
                rest_type = "distance"
                data = (md.compute_distances(traj, [index_list])*10).flatten() 
                data_list.append(data)
                label_list.append(window)
                if not decoration:
                    unit_str = r" $\AA$"
                    plt.xlabel(r"Distance ($\AA$)")
                    plt.title(f"Distance between {mask_list[0]} and {mask_list[1]}" + "\n" + f"in phase \"{phase_name}\"\n" + "\n" + ("Static restraint" if static_rest else "Dynamic restraint"))

            elif len(index_list) == 3:
                rest_type = "angle"
                data = (md.compute_angles(traj, [index_list])*180/np.pi).flatten()  # To degrees
                if not shift_to_avoid_split:
                    fraction_at_extremes = np.mean((data < -170)) + np.mean((data > 170))
                    if fraction_at_extremes >= 0.75:
                        shift_to_avoid_split = True
                data_list.append(data)
                label_list.append(window)
                if not decoration:
                    unit_str = r"$^{\circ}$"
                    plt.xlabel(r"Angle ($^{\circ}$)")
                    plt.title(f"Angle between {mask_list[0]}, {mask_list[1]}, and {mask_list[2]}" + "\n" + f"in phase \"{phase_name}\"" + "\n" + ("Shifted to avoid split!\n" if shift_to_avoid_split else "\n") + ("Static restraint" if static_rest else "Dynamic restraint"))

            elif len(index_list) == 4:
                rest_type = "torsion"
                data = (md.compute_dihedrals(traj, [index_list])*180/np.pi).flatten()  # To degrees
                if not shift_to_avoid_split:
                    fraction_at_extremes = np.mean((data < -170)) + np.mean((data > 170))
                    if fraction_at_extremes >= 0.75:
                        shift_to_avoid_split = True
                data_list.append(data)
                label_list.append(window)
                if not decoration:
                    plt.xlabel(r"Torsion ($^{\circ}$)")
                    unit_str = r"$^{\circ}$"
                    plt.title(f"Torsion between {mask_list[0]}, {mask_list[1]}, {mask_list[2]}, and {mask_list[3]}" + "\n" + f"in phase \"{phase_name}\"" + "\n" + ("Shifted to avoid split!\n" if shift_to_avoid_split else "\n") + ("Static restraint" if static_rest else "Dynamic restraint"))

        
            temp_minimum, temp_maximum = min(data), max(data)
            if temp_minimum < minimum:
                minimum = temp_minimum
            if temp_maximum > maximum:
                maximum = temp_maximum

            decoration = True
            
        
        if rest_type in ["angle", "torsion"] and shift_to_avoid_split:
                # target += 180
                data_list = [[d+360 if d<=0 else d for d in data] for data in data_list]  # Avoid split-up visualization by mapping (-180,180) to (0,360)
                minimum = min([min(data) for data in data_list])
                maximum = max([max(data) for data in data_list])

        bins = np.linspace(minimum, maximum, 500)
        # colors = ["blue", "orange", "yellow", "purple"]
        # colors = plt.get_cmap('tab10').colors
        okabe_ito = [
            "#E69F00", "#56B4E9", "#009E73", "#F0E442",
            "#0072B2", "#D55E00", "#CC79A7", "#000000"
        ]
        for i, data in enumerate(data_list):
            if i < 5 or True:
                plt.hist(data, bins=bins, label=label_list[i], alpha=1.0, histtype='step', linewidth=2, color=okabe_ito[i])
            else:
                plt.hist(data, bins=bins, alpha=0.4)


        old_ylim = plt.ylim()
        plt.vlines(target, 0, 1e5, color="black", label=f"Target: {target:.2f}{unit_str}")
        plt.ylim(old_ylim)
        
        plt.ylabel("Count")
        plt.legend()
        if not os.path.exists(f"{results}/{rest_type}"):
            os.mkdir(f"{results}/{rest_type}")
        plt.tight_layout()
        plt.savefig(f"{results}/{rest_type}/{counter}_{rest_type}.png", format="png")
        plt.savefig(pdf, format="pdf")

    pdf.close()
            