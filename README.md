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

## matlab instruction for lewos

- Install matlab
- Install required toolboxes (home tab add-ons icon)
<img width="223" alt="add-ons" src="https://user-images.githubusercontent.com/36380548/152797738-39fb2f45-9e1a-4f94-861d-390a11575e4e.png">

- - Text Analytics ToolBox
- - Deep Learning Toolbox
- - Statistics and Machine Learning Toolbox
- - Lidar Toolbox
- - Image Processing Toolbox
- - Computer Vision Toolbox
- add `tile_10_-10.las` to LeWoS folder
- add to LeWoS folder new function loadDataPoints.m with following code and run it:  
```
lasReader = lasFileReader("tile_10_-10.las");
ptCloud = readPointCloud(lasReader);
allRows = ptCloud.Location(:,:);
```
- In workspace new variable `allRows` will appear
- Add to LeWoS folder new function pickanameyourself.m with following code and run it:
```
[BiLabel, BiLabel_Regu] = RecursiveSegmentation_release(allRows, 0.135, 0, 1)
```

