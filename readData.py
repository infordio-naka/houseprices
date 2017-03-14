import pandas       as pd
import scipy.stats  as sp
import numpy        as np

class ReadData(object):
    def __init__(self):
        pass

    def readCSV(self, path):
        df = pd.read_csv(path)
        return (df)

    def readTSV(self, path):
        df = pd.read_csv(path, delimiter='\t')
        return (df)

    def getCol(self, df, colList):
        col = df.loc[:, colList]
        return (col)

    def preprocess(self, df, colList):
        p_df = df.fillna(df.mean())
        p_df = pd.DataFrame(sp.stats.zscore(p_df), columns=colList) # normalization
        return (p_df)
        
if __name__ == "__main__":
    rd      = ReadData()
    df      = rd.readCSV("dataset/train.csv")
    colList = ["LotFrontage", "SalePrice"]
    data    = rd.getCol(df, colList)
    data    = rd.preprocess(data, colList)
    print(data)
    x_data  = rd.getCol(data, colList[:-1])
    y_data  = rd.getCol(data, colList[1:])
    print(np.asarray(x_data))
    print(np.asarray(y_data))
