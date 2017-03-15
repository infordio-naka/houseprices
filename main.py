import os
import gc
import numpy    as np
import pandas   as pd
from   readData import ReadData
from   model    import Model

def main(args):
    # config
    batch_size = 10
    nb_epoch   = 1000
    train_file = "dataset/train.csv"
    pred_file  = "dataset/test.csv"
    colList    = ["LotFrontage",
                  "MSSubClass",
                  "LotArea",
                  "YearRemodAdd",
                  "MasVnrArea",
                  "BsmtFinSF1",
                  "BsmtUnfSF",
                  "TotalBsmtSF",
                  "1stFlrSF",
                  "BedroomAbvGr",
                  "SalePrice"] # [x1, x2, ..., xn, y]
    save_dir   = "./result/"

    # read dataset
    rd       = ReadData()
    train_df = rd.readCSV(train_file)
    pred_df  = rd.readCSV(pred_file)
    
    t_data = rd.getCol(train_df, colList)
    t_data = rd.preprocess(t_data, colList)
    p_data = rd.getCol(pred_df, colList)
    p_data = rd.preprocess(p_data, colList)
    p_data = np.asarray(rd.getCol(pred_df, colList[:-1]))

    x_data = np.asarray(rd.getCol(t_data, colList[:-1]))
    y_data = np.asarray(rd.getCol(t_data, [colList[-1]]))

    # create model
    md = Model()
    input_shape  = x_data.shape
    output_shape = 1
    md.create_model(input_shape, output_shape)

    # train
    """
    md.train(x_data, y_data, batch_size, nb_epoch, verbose=1)
    md.save(save_dir, save_dir)
    """

    # predict
    md.predict(p_data, save_dir)

if __name__ == "__main__":
    args = None
    main(args)
    gc.collect()
