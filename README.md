# SPADE_surrogates
Repository containing code necessary to reproduce the results of Stella, A., Bouss, P., Palm, G., &amp; Grün, S. (2021). Generating surrogates for significance estimation of spatio-temporal spike patterns

**Cloning the repo**

Please clone the repository by running:
`git clone --recurse-submodules git@github.com:INM-6/SPADE_surrogates.git`.

Alternatively, you can proceed by:

`git clone git@github.com:INM-6/SPADE_surrogates.git`

`git submodule update --init`.


In order to download the data, please, go into the multielectrode_grasp folder 
by typing:
`cd data/multielectrode_grasp`.

Continue with `gin login`.

If you don't have a gin-account or client, refer to the README in the multielectrode_grasp folder.

Then you need to run:
`gin get-content i140703-001-03.nev l101210-001-02.nev i140703-001.odml l101210-001.odml`. This will download the
results of the spike-sorting (.nev) and metadata files (.odml). 

**Creating the conda environment**

For general instructions on how to use the conda environments, please refer to:
https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html. 

Create the environment by running: `conda env create -f env.yml`

There might be a problem to install `mpi4py`. In this case, it would be necessary 
to first install OpenMPI manually, e.g., on a Linux machine you can use `sudo apt install libopenmpi-dev`.

Switch to the environment: `conda activate surrogates`

Integrate this project as editable package: `pip install -e .`


**Creating the figures**

Please go to the code folder `cd code`
- For Figure 2:
  - `python fig2_spikeloss_r2gstats.py`  (This takes some time.)
- For Figure 3, Panel B: 
  - `python fig3_panelB_spike_loss.py`
- For Figure 3, Panel C:
  - `cd fig3_panelC_pvalue_spectrum`
  - `python max_order_statistics.py`
  - `python create_independent_spiketrains.py`
  - `mpirun python analyze_independent_spiketrains.py` (This step should be done on a cluster)
  - `python plot_pvalue_spectrum.py`
  - `cd ..`
- For Figure 5:
  - `python fig5_surrogate_statistics_data.py` (This takes some time.)
  - `python fig5_surrogate_statistics_plot.py`
- For Figure 6:
  - `python fig6_r2gexperiment.py`
- For Figure 7:
  - `cd analysis_artifial_data`
  - `snakemake`  (This step should be done a cluster.)
  - `cd ..`
  - `python fig7_artificial_data.py`
- For Figure 8:
  - `cd analysis_experimental_data.py`
  - `snakemake`  (This step should be done a cluster.)
  - `cd ..`
  - `python fig8_experimental_data.py`
- For S1 Figure:
  - `python S1_fig_cv2s_in_fps.py`  (This takes some time.)
- For S2 Figure:
  - download and install  https://borgelt.net/src/pycoco.zip 
  - `python S2_fig_coconad.py`