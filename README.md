# AO_Erosion_Simulator
Atomic-oxygen erosion simulator for LEO spacecraft surfaces. Couples molecular dynamics-derived local erosion data with a level-set PDE and a statistical, poly-Gaussian multi-reflection flux model to predict long-term surface morphology and global erosion yield evolution under hyper-thermal flow conditions.

## Installation

### 1) Get the code

Clone the repository and enter it:

```bash
git clone https://github.com/sabinanton/AO_Erosion_Simulator.git
cd AO_Erosion_Simulator
```

### 2) Create a clean Python environment (recommended)

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

This project requires **Python 3.10+**.

Core dependencies:

* `numpy` (2.0+ recommended)
* `scipy`
* `matplotlib`

Install them with:

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib
```

Notes:

* `os`, `argparse`, `typing`, and `datetime` are part of the Python standard library and do not need to be installed.
```

If this fails with `ModuleNotFoundError`, double check that `src/` is on your `PYTHONPATH` and that you are running commands from the repository root.


## Usage

From the repository root, run:

```bash
python3 src/erosion_simulator.py "<PATH_TO_PROJECT_FOLDER>"
```

Where `<PATH_TO_PROJECT_FOLDER>` is the folder that contains `Inputs.txt`.

Example:

```bash
python3 src/erosion_simulator.py projects/test_project
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

Licensed under the [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/).
