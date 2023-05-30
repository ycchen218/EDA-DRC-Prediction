# DRC-Prediction
## Introduce
This is a deep-learning-based model used to predict the location of DRC Violation.The predict location violate one of the rule as following:

|  Violation Rule        | Violation Rule|Violation Rule|
| :-------------: |:-------------:| :-----:|
| AdjacentCutSpacing        | CornerFillSpacing      |CutEolSpacing |
| CutShort        | DifferentLayerCutSpacing      |Enclosure |
| EnclosureEdge      | EnclosureParallel      |    EndOfLineSpacing|
| FloatingPatch      | JogToJogSpacing      |    MaximumWidth|
| MaxViaStack      | MetalShort      |    MinHole|
| MinimumArea      | MinimumCut      |    MinimumWidth|
| MinStep      | Non-sufficientMetalOverlap      |    NotchSpacing|
| OutOfDie      | ParallelRunLengthSpacing      |    SameLayerCutSpacing|


## Requirement
1. python3.8
2. scipy
3. matplotlib
4. numpy
5. pytorch 1.12.0
6. pandas
7. scipy
## Predict
```markdown
python drc_predict.py
```
--data_path: The path of the data file <br>
--fig_save_path: The path you want to save figure <br>
--weight_path: The path of the model weight <br>
--output_path: The path of the predict output with .npy file and .csv file <br>
--drc_threshold: drc_threshold [0,1] <br>
--device: If you have gpu type "cuda" will be faster!! <br>
## Predict result
1. Tune your own drc_threshold, the defalt is 0.001 as shown in following figure.
![image](https://github.com/ycchen218/DRC-Prediction/blob/master/git-image/DRC_0.001.png)
2. drc_threshold = 0.1 <br>
![image](https://github.com/ycchen218/DRC-Prediction/blob/master/git-image/DRC_0.01.png)
## Compare with ground truth
![image](https://github.com/ycchen218/DRC-Prediction/blob/master/git-image/compare.png)
## Cross validation while evalulate the model
We achieved the AUC with 0.99 with threshold=0.1 . <br>
by the metrics code are same as [CircuitNet](https://github.com/circuitnet/CircuitNet)
