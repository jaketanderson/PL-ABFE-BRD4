# Protein-Ligand Absolute Binding Free Energy of the First BRD4 Bromodomain

## Background

This repository contains one or more notebooks to calculate the free energy of binding of one or more ligands to the first BRD4 bromodomain. My goal is to evaluate an OpenFF `ForceField` containing a new `CustomGBSA` potential, but the code could be adapted for other use cases requiring explicit solvent.

The setup process will avoid AMBER as much as possible, as outlined in [this tutorial notebook](https://github.com/GilsonLabUCSD/pAPRika/blob/master/docs/tutorials/08-tutorial-sug-roc-notleap.ipynb). The specifics of the evaluation will come from Germano Heinzelmann, Neil Henriksen, and Michael Gilson's 2017 paper on this exact simulation. [1] Where I deviate from the specifications of that paper, such as by altering a force constant or numbers or simulation times, I will try to make it clear.

[1] Heinzelmann, G.; Henriksen, N. M.; Gilson, M. K. Attach-Pull-Release Calculations of Ligand Binding and Conformational Changes on the First BRD4 Bromodomain. Journal of Chemical Theory and Computation, 2017, 13, 3260–3275. [https://doi.org/10.1021/acs.jctc.7b00275](https://doi.org/10.1021/acs.jctc.7b00275)

## Warning
I'm using some customized packages in my anaconda environment, so things may be broken if you run these notebooks out of the box. There is a separate repository containing the alterations and a script that automatically installs them, but the repository is private because the project it was written for is with an industry partner. If anybody wants to use the code in this repository and I haven't yet made public a way to get the customized packages, please let me know and I'd be happy to help.

## File structure
| File/directory | Purpose |
|--|--|
| `initial_data/` | Unprocessed structure and charge files.|
| `prepared_data/` | Processed structure and charge files. |
| `working_data/` | This directory will start off empty and will be populated <br> with directories for each window being simulated. |
