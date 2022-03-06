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


## instruction for cloud compare
* First download and install CloudCompare https://www.danielgm.net/cc/  
* Open CloudCompare and drop file (from puhti path LeafAndWood/data/labeled/) to main window.
<img width="1006" alt="Näyttökuva 2022-3-6 kello 20 19 22" src="https://user-images.githubusercontent.com/36380548/156937601-fb4744cb-cf7f-478c-9c3a-4f05317d758b.png">   

* Select tile, go to properties window, in scalar field secition choose "Scalar Field #5"
<img width="1280" alt="Näyttökuva 2022-3-6 kello 20 20 24" src="https://user-images.githubusercontent.com/36380548/156937924-90f5e8fd-7d7b-4f2e-92f0-9431ccc8393f.png">  

* After that go to "Edit" -> "Scalar fields" -> "Filter by value". After that in pop up window set range from 0 to 0 (!)  

<img width="1280" alt="Näyttökuva 2022-3-6 kello 20" src="https://user-images.githubusercontent.com/36380548/156938132-7271327d-1e54-4102-9228-950400ff305f.png">  

<img width="1280" alt="Näyttökuva 2022-3-6 kello 20 21 39" src="https://user-images.githubusercontent.com/36380548/156938146-b2cf7139-9bbb-4ab6-80b0-2d760977be38.png">  

* Now you will have two different clouds. "Cloud.extract" is all leafs and "Cloud.extract.outside" is all woods. Select only Cloud.extract.outside and make sure to pick "Scalar Field #5" in Properties scalar field secition.  
* After that select only "Cloud.extract.outside" and go to "Tools" -> "Segmentation" -> "Label Connected Comp." and agree with default settings  
<img width="1280" alt="Näyttökuva 2022-3-6 kello 20 23" src="https://user-images.githubusercontent.com/36380548/156938362-782afb01-25b4-4382-837d-a10cee981edd.png">  

* Now points will be partitioned to smaller clouds  
<img width="1162" alt="Näyttökuva 2022-3-6 kello 20 23 55" src="https://user-images.githubusercontent.com/36380548/156938418-4da084ab-9389-4668-8825-1bb1498b1b97.png">  
 
* Now it is posible to select those parts that clearly should be leafs by clicking on them  
<img width="1280" alt="Näyttökuva 2022-3-6 kello 20 24" src="https://user-images.githubusercontent.com/36380548/156938489-8387d3c8-3d0d-4333-a5bc-4d9a4fd9b527.png">  

* If you want to change certain group from wood to leaf, select it and go to "Edit" -> "Scalar fields" -> "Delete". This step deletes currently active scalar field ("Scalar Field #5") for that particular group, but does not remevo the group.
<img width="1280" alt="Näyttökuva 2022-3-6 kello 20 24 51" src="https://user-images.githubusercontent.com/36380548/156938628-0c3acd7a-4c60-414b-a74c-78d3b19304a2.png">   

* After that add new scalar field to selecter group of points (on picture plus icon). In the example I give the same name "Scalar Field #5" and assign the value 0 (leaf).
<img width="1280" alt="Näyttökuva 2022-3-6 kello 20 25 12" src="https://user-images.githubusercontent.com/36380548/156938744-c7b6ace6-9b3e-4abc-b24d-8d35d22d1a6c.png">   

* Repeat the process for those groups that you see as leafs
<img width="234" alt="Näyttökuva 2022-3-6 kello 20 34 38" src="https://user-images.githubusercontent.com/36380548/156938810-bdd7d9ac-5964-4204-9059-58994c38e60c.png">  

* After that, select all smaller clouds (with shift button) and push "Merge multiple clouds button"
<img width="877" alt="Näyttökuva 2022-3-6 kello 20 35 03" src="https://user-images.githubusercontent.com/36380548/156938904-4006464d-7c90-4812-b0ba-dec6f9294b07.png">  

<img width="99" alt="Näyttökuva 2022-3-6 kello 21 26" src="https://user-images.githubusercontent.com/36380548/156938994-15b103c8-94a0-44a0-9e38-7f89afc55198.png">  

* Now pick-n-drop combined cloud "CC#0" to original folder together with "Cloud.extract"  
<img width="430" alt="Näyttökuva 2022-3-6 kello 20 36 52" src="https://user-images.githubusercontent.com/36380548/156939118-725a6d26-6fee-4cdc-a712-b141e12fdb43.png">  

* Now select both of them and merge once again
* Done! Don't forget to save. You can compare results but opening the original once again 
<img width="1197" alt="Näyttökuva 2022-3-6 kello 20 39 59" src="https://user-images.githubusercontent.com/36380548/156939209-d3a2082e-49a1-4417-a1e9-12a68b18b739.png">  

<img width="1196" alt="Näyttökuva 2022-3-6 kello 20 40 06" src="https://user-images.githubusercontent.com/36380548/156939211-477663f1-f586-47d3-9adf-b9331ffdc29d.png">  

## matlab instruction for lewos
- Install matlab
- Install required toolboxes (home tab add-ons icon)
<img width="223" alt="add-ons" src="https://user-images.githubusercontent.com/36380548/152797738-39fb2f45-9e1a-4f94-861d-390a11575e4e.png" >

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

