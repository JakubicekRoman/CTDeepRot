import pandas as pd 
import numpy as np


xl_file = pd.ExcelFile("D:/jakubicek/Rot_detection/data/training/labels_bin.xlsx")




data = pd.read_excel(xl_file,header=None)


file_names=data[0].values.tolist()

labels=data.loc[:,1:7].to_numpy()






