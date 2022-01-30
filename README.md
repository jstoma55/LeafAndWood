# LeafAndWood

## Dependencies

Install dependencies to virtual Python environment, this is so we all have the same package versions.

1. `python3 -m venv env`
2. `source env/bin/activate`
3. `python3 -m pip install -r requirements.txt`

If you use some package please add it to the `requirements.txt`. You can get installed packages and their version by using `python3 -m pip freeze`

## Samples 
- There is a sample .las file in `data/samples`
- Example Jupyter Notebook

## Structure

- `data` for all the data used for project
- `notebooks` for Jupyter Notebooks
- `reports` for reporting model performance, etc.
- `src` all the classes, methods for model development, file manipulation, data processing, visualization

