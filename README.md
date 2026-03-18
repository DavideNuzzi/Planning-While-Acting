# Planning-while-acting: addressing the continuous dynamics of planning and action in a virtually embodied task

This repository contains the code, data, and analysis notebooks for the paper **"Planning-while-acting: addressing the continuous dynamics of planning and action in a virtually embodied task"** (Nuzzi, Cisek, & Pezzulo, PNAS).

In this study, we investigate how people make sequential embodied decisions by balancing immediate affordances (e.g., safe vs. risky jumps) with long-term utility (path length) in a virtually embodied environment. 

## Play the Experiments Online

The experiments were developed as 3D video games in Unity. You can play the exact versions of the tasks used in the study directly in your browser via GitHub Pages:

* **[Main Experiment](https://davidenuzzi.github.io/Planning-While-Acting/Experiment_V1/index.html)** The core river-crossing task. Players control a frog avatar and must choose between crossing larger, closer rocks ("safe" jumps) and smaller, farther rocks ("risky" jumps) to reach the red flag.
* **[Control Experiment 1: Delayed Movement](https://davidenuzzi.github.io/Planning-While-Acting/Experiment_V2/index.html)** To test the influence of physical momentum, the avatar's movement is blocked for 1 second after landing on each rock. You can rotate the camera, but cannot immediately jump.
* **[Control Experiment 2: Delayed Movement + Forced Rotation](https://davidenuzzi.github.io/Planning-While-Acting/Experiment_V3/index.html)** In addition to the 1-second delay upon landing, the avatar is automatically rotated at decision points to face the direction of the "new" (alternative) jump path.

**Controls:**
* **Mouse:** Adjust the camera view.
* **W:** Move forward.
* **Spacebar:** Jump.

---

## Data and Code Structure

The raw data from the experiments is stored as `.pkl` files inside the `data` folder. This data is structured using custom Python classes that collect all the relevant information about the game, the levels, and the players' choices. 

To understand how this data is formatted, you should read the `CrossTheRiver.py` file. It contains all the class definitions as well as the useful functions needed to work with the datasets. 

To reproduce the analyses and the figures presented in the paper, simply run the corresponding Jupyter Notebooks which show exactly how each figure was generated.

---

## Environment Setup and Installation

This project uses specific statistical packages for its linear mixed model analyses. Because of strict version dependencies, **we highly recommend using Conda** to set up the environment.

1.  **System Requirements:** You must have **R** installed on your system, as the linear mixed models depend on it.
2.  **Conda Environment:** Use the provided `environment.yml` file to recreate the exact environment. This will install a specific version of `pymer4`, which is strictly required to replicate our LMM analyses.

```bash
# Clone the repository
git clone [https://github.com/davidenuzzi/Planning-While-Acting.git](https://github.com/davidenuzzi/Planning-While-Acting.git)
cd Planning-While-Acting

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate crosstheriver