
# A Regression Mixture Model to understand the effect of the Covid-19 pandemic on Public Transport Ridership
This repository contains the code for two publications that are currently under review. You will find all the code for preprocessing, parameter estimation and figure display.

## Code specificities 
The code does several things the publication presents differently/does not do:
- Another dataset is present for anyone to use: the CER dataset, downloadable at  . This dataset has limited interest to us because the data do not present clear segments. 
- The exogenous variables (denoted $X$ in the publication) are split into three types of variables, to avoid replicating values:
	- `individual_variables`, such that $x_{i,t} = x_{i,1}$, such as the presence of RER train in each station.
	- `temporal_variables`, such that $x_{i,t} = x_{1,t}$, such as public holidays or lockdowns.
	- `mixed_variables`, which depend on both $i$ and $t$. We did not have any such variables for the IdFM dataset.
	
- Similarly, the regression coefficients ($\alpha$ in the publication) are split into `alpha`, `beta`, `gamma` for individual, temporal, and mixed variables, respectively.
-  The constraint $\lambda$ has been renamed `min_slope`. 
-  The covariance matrix $\Sigma_{k,s}$ is named `sigma` (the lower case s does not denote a standard deviation but a variance).



## Dependencies
All libraries are listed in `environment.yml`:  `conda create --name [envname] --file environment.yml` . 
Note that one file (`visualizations/location_data_IdFM.py`) uses geopandas and contextly, which in turn depend on system libraries such as GDAL.

## How to use: 
- Download the necessary data: 
	- IdFM rail data at https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-ferre/ and https://data.iledefrance-mobilites.fr/explore/dataset/validations-reseau-ferre-nombre-validations-par-jour-1er-semestre/information/
	- list of public holidays: https://ardennemetropole.opendatasoft.com/explore/dataset/jours-ouvresweek-endferies-france-2010-a-2030/
	- [optional] IdFM bus entries at https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-surface/information/, https://data.iledefrance-mobilites.fr/explore/dataset/validations-reseau-surface-nombre-validations-par-jour-1er-trimestre/information/, and https://data.iledefrance-mobilites.fr/explore/dataset/validations-reseau-surface-nombre-validations-par-jour-2eme-trimestre/information/
	- [optional] the possibility of using the weather is here to see whether Parisians travel more on sunny days: https://www.historique-meteo.net/site/export.php?ville_id=188
- Write the location of the data in your computer in `preprocessing/IdFM/data_path.py` 
- The experiments can now be launched from the `main_clustering_TRB.py` or `clustering_results_ICDM.py` scripts. The first run of `preprocessing/dIdFM/load_data.py` takes a few hours, but it will store some preprocessed data in a `.pickle` file to save time on later uses. 

Most files also contain unit tests or basic displays that can be accessed by executing them directly (as opposed to importing them). Feel free to explore and experiment !


## Contents
The code is organized as follows:
- **preprocessing/IdFM** contains the structures to load the raw data and exogenous variables \(`load_raw_data.py`\), and preprocess them \(`load_data.py`\), along with hardcoded dates of strikes and school holidays. 
- **models** contains all the codes implementing the proposed model  \(`main_model.py`\)  along with two baselines that do not appear in the publication \( `baseline_no_time.py` and `SegClust.py` \). Note that even though SegClust comes from another publication, the code is a third party (unofficial) implementation. 
- **experiments** contains `main_clustering.py`, which serves to display most figures from the paper. `compare_performance_synthetic_data.py` and `compare_performance_real_data.py` generate the results we see in tables I and III, respectively. 
	- **model_selection** contains both an unofficial implementation of the slope heuristic and the code to  generate many models with different number of clusters.  
- **utils** gathers auxiliary scripts that reimplement linear and logistic regressions the way we want it, along with other methods we use throughout the code.
- **visualizations** contains scripts allowing result display and interpretation: 
	- `clustering_results_TRB.py` and `clustering_results_ICDM.py` (one file for each publication) implement generic plots.
	- `location_data_IdFM.py` shows the location of the stations on a Paris map, and assigns each station a color that depends on the cluster.
	- `model_similarity.py` compares the cluster/segment assignment of couples of models, and displays the Adjusted Rand Index allowing to measure how similar are the predictions of two models. 
	- similarly, `cluster_similarity.py` takes the couples of $(individual, timesteps)$ assigned by one model to each cluster/segment and tries to see whether they fit in other clusters/segments. 
	- `time_series.py` implements a line plot with lines of varying color.  
	- `significant_difference.py` implements a t-test to know which regression coefficients change between clusters or segments. 



## Results

<img src="results/figures/TRB/IdFM_data_example.pdf" alt="Some normalized series" width="600"/>

<img src="results/figures/TRB/cluster_locations_IdFM_zoomed.pdf" alt="The location of each cluster inside Paris" width="600"/>

<img src="results/figures/TRB/reconstruction_splines.pdf" alt="The effect of time on each segment" width="600"/>


<img src="results/figures/TRB/IdFM_effect_variables.pdf" alt="The regression coefficients associated to days of the week" width="600"/>



## Reference
[tbd]

