# GridCAT Python

This repository contains a work-in-progress Python translation and exploration of the MATLAB GridCAT Toolbox for fMRI grid-code analysis. 

The current repository contains two notebooks:

- `Example_with_Comparison.ipynb` contains the full setup and comparison workflow, including checks against Matlab/GridCAT reference outputs.
- `Examples.ipynb` contain a clean example workflow only, without the extended comparison and development diagnostics.

The comparison notebook is intended to make the implementation easier to inspect, validate, and iterate on. The example workflow notebook is intended to provide a smaller entry point once the comparison-heavy development notebook becomes too noisy for ordinary use.

## Status

This implementation is currently working for the included/example workflow and was validated on additional data (Graichen et al., 2025), however this project is still under active development. Assumptions, outputs, and notebook structure may change without notice.

This code is provided "as is", without warranty of any kind, express or implied, including but not limited to warranties of correctness, fitness for a particular purpose, reproducibility, or non-infringement. Use it at your own risk. The author is not liable for any claim, damages, data loss, incorrect scientific conclusions, or other liability arising from use of this code.

## Relationship to GridCAT

This is a personal project and is not affiliated with, endorsed by, or maintained by the authors of the original GridCAT toolbox.

If you use or discuss this work, please cite the original GridCAT publication where appropriate:

Stangl, M.*, Shine, J.*, & Wolbers, T. (2017). The GridCAT: A toolbox for automated analysis of human grid cell codes in fMRI. *Frontiers in Neuroinformatics, 11*, 47. [* equal contribution]  
https://doi.org/10.3389/fninf.2017.00047

## Repository Contents

- `gridcat.py` contains the main Python implementation.
- `Compare.py` contains helper functionality for comparing Python outputs against Matlab/GridCAT outputs.
- `Example_with_Comparison.ipynb` contains the comparison-oriented notebook workflow.
- `Example.ipynb` contains the clean example workflow.


## Notes

This repository is currently best understood as a development and validation workspace rather than a polished software package. Please review the notebooks and source code carefully before relying on any results.