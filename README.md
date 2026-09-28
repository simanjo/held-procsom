# held-procsom
Code and Plots accompanying the publication "HEterogeneous Literature Data limit machine-learning PRediction of Organic Contaminant SOrption to Microplastics".

Data preparation - from a JSON like format to a pandas DataFrame - is accomplished with the function [save_raw_data()](https://github.com/simanjo/held-procsom/blob/0b6227c862290377f61a29be2dd2449d50b0aea4/preprocessing/data.py#L213) in the [preprocessing module](preprocessing/data.py). All further analysis expects csv files in the format specified there.

Model training is accomplished by executing the [training script](training/training.py), which expects the data to be stored in a data subfolder (adapt [DATA_DIR](https://github.com/simanjo/held-procsom/blob/0b6227c862290377f61a29be2dd2449d50b0aea4/training/training.py#L65) if necessary). For the addition of the abrahams parameters, a csv file named RMG_SoluteML.csv is required, with a format specified by the output of the [Reaction Mechanism Generator](https://rmg.mit.edu/database/solvation/soluteSearch/) with SoluteML choosen as model (cf [source code](https://github.com/simanjo/held-procsom/blob/0b6227c862290377f61a29be2dd2449d50b0aea4/preprocessing/embeddings.py#L168) for details).
Training of the [Leave-one-Group-out models](training/leaveXout.py) and the [mixed effect models](training/grouped_training.py) for nested leave-one-publication-out were accomplished by the linked scripts.

The [plots](plots/) in paper and supplement were produced with the notebooks for the [descriptive analysis](analysis/descriptive_analysis.ipynb) of the dataset, the [model evaluation](analysis/plots.ipynb) plots and the [supplemental tables](analysis/suppl-tables.ipynb). The permutation tests were performed with a [notebook](analysis/permutation_tests.ipynb). [Additional plots](plots/extra_plots) similar to the ones in the manuscript/supplemental material were produced with the notebook for [extra plots](analysis/extra_plots.ipynb).

