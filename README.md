# Genetic Algorithm Selection of Interacting Features (GASIF) for Selecting Biological Gene-Gene Interactions

This repository hosts the code for GASIF (Genetic Algorithm Selection of Interacting Features), as published in the Proceedings of the Genetic and Evolutionary Computation Conference (doi: https://doi.org/10.1145/3638529.3654159).


## Installation

The following dependencies are required:

```
numpy
pandas
sklearn
ipdb
joblib
```

All of these dependencies can be installed using ```pip``` or ```conda```. 

```ipdb``` is required as a dependency used for interactive, in-line debugging, but you can alternatively comment out all of the ```ipdb``` imports from the files if you would prefer not to install this dependency.


## Execution

We've included a sample script ```main.py``` that imports GASIF and runs it as required for reference.

You should edit the data paths on [lines 95-97](https://github.com/rachitk/biofeature-interaction-selection-ga/blob/main/main.py#L95-L97) to run GASIF with your own data

You can also edit the running options on [lines 31-48](https://github.com/rachitk/biofeature-interaction-selection-ga/blob/main/main.py#L31-L48) to run GASIF with different options. 
