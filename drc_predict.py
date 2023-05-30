import os
import argparse
import numpy as np
import torch
from scipy import ndimage
from drc_model import DRCModel
import matplotlib.pyplot as plt
import pandas as pd

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DRCPrediction():
    def __init__(self,datapath,features,model_weight_path,device):
        super(DRCPrediction, self).__init__()
        self.datapath = datapath
        self.FeaturePathList = features
        self.feature = self.data_process(self.FeaturePathList).unsqueeze(0).to(device)
        self.model = DRCModel(device).to(device)
        self.device = device
        checkpoint = torch.load(model_weight_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def resize(self,input):
        dimension = input.shape
        result = ndimage.zoom(input, (256 / dimension[0], 256 / dimension[1]), order=3)
        return result

    def std(self,input):
        if input.max() == 0:
            return input
        else:
            result = (input - input.min()) / (input.max() - input.min())
            return result

    def data_process(self,FeaturePathList):
        features = []
        for feature_name in FeaturePathList:
            name = os.listdir(os.path.join(self.datapath,feature_name))[0]
            feature = np.load(os.path.join(self.datapath,feature_name,name))
            feature = self.std(self.resize(feature))
            features.append(torch.as_tensor(feature))
        features = torch.stack(features).type(torch.float32)
        return features

    def find_drc_coord(self,tensor, threshold):
        p = torch.where(tensor>=threshold,1,0)
        indices = torch.where(p == 1)
        return np.array(list((indices[1].tolist(), indices[0].tolist()))).T

    def Prediction(self, drc_threshold):
        self.drc_threshold = drc_threshold
        if self.device != 'cpu':
            with torch.cuda.amp.autocast():
                self.pred = self.model(self.feature)
                self.pred = self.model.sigmoid(self.pred)
        if self.device == 'cpu':
            self.pred = self.model(self.feature)
            self.pred = self.model.sigmoid(self.pred)
        self.pred_coord = self.find_drc_coord(self.pred[0,0], threshold=drc_threshold)
        self.pred_coord = pd.DataFrame(self.pred_coord,columns=['x','y'])
        return self.pred, self.pred_coord

    def ShowFig(self,fig_save_path):
        if fig_save_path is None:
            raise ValueError("Figure save path is not specified clear.")
        plt.imshow(self.pred[0, 0].detach().cpu().numpy())
        plt.title(f"DRC Prediction")
        pts = plt.scatter(x=self.pred_coord['x'],y=self.pred_coord['y'],c='r',s=5)
        plt.legend([pts],["Violation"])
        plt.savefig(f"{fig_save_path}/DRC_{self.drc_threshold}.png")
        plt.show()

    def save(self,output_path):
        np.save(f"{output_path}/PredArray",self.pred[0,0].detach().cpu().numpy())
        self.pred_coord.to_csv(f"{output_path}/PredCoord.csv")


def parse_args():
    description = "Input the Path for Prediction"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_path", default="./data", type=str, help='The path of the data file')
    parser.add_argument("--fig_save_path", default="./save_img", type=str, help='The path you want to save fingue')
    parser.add_argument("--weight_path", default="./model_weight/drc_weights.pt", type=str, help='The path of the model weight')
    parser.add_argument("--output_path", default="./output", type=str, help='The path of the model weight')
    parser.add_argument("--drc_threshold", default=0.001, type=float, help='drc_threshold [0,1]')
    parser.add_argument("--device", default='cpu', type=str, help='If you have gpu type "cuda" will be faster!!')
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = parse_args()
    feature_list = ['macro_region', 'cell_density','RUDY_long', 'RUDY_short','RUDY_pin_long',
        'congestion_eGR_horizontal_overflow',
        'congestion_eGR_vertical_overflow',
        'congestion_GR_horizontal_overflow',
        'congestion_GR_vertical_overflow']

    predictionSystem = DRCPrediction(datapath=args.data_path,features=feature_list,
                                model_weight_path=args.weight_path,device=args.device)
    pred,pred_coord = predictionSystem.Prediction(drc_threshold=args.drc_threshold)
    predictionSystem.save(args.output_path)
    if args.fig_save_path !=None:
        predictionSystem.ShowFig(args.fig_save_path)

