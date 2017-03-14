import os
import gc
import numpy    as np
from   readData import ReadData
from   model    import Model

def main(args):
    # config
    batch_size = 10
    nb_epoch   = 1000
    train_file = "dataset/train.csv"
    colList    = ["LotFrontage",
                  "MSSubClass",
                  "LotArea",
                  "YearRemodAdd",
                  "MasVnrArea",
                  "BsmtFinSF1",
                  "BsmtUnfSF",
                  "TotalBsmtSF",
                  "1stFlrSF",
                  "SalePrice"] # [x1, x2, ..., xn, y]
    save_dir   = "./"

    # read dataset
    rd = ReadData()
    df = rd.readCSV(train_file)
    data = rd.getCol(df, colList)
    data = rd.preprocess(data, colList)
    x_data = np.asarray(rd.getCol(data, colList[:-1]))
    y_data = np.asarray(rd.getCol(data, [colList[-1]]))

    # create model
    md = Model()
    input_shape  = x_data.shape
    output_shape = 1
    md.create_model(input_shape, output_shape)

    # train
    md.train(x_data, y_data, batch_size, nb_epoch, verbose=1)
    md.save(save_dir)

if __name__ == "__main__":
    args = None
    main(args)
    gc.collect()
