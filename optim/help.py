import numpy as np
from  sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
import pandas as pd

def softmax(x):
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

def RescaledSaliency(mask, isTensor=True):
    if(isTensor):
        saliency = np.absolute(mask.data.cpu().numpy())
    else:
        saliency = np.absolute(mask)
    saliency  = saliency.reshape(mask.shape[0], -1)
    rescaledsaliency = softmax(saliency)
    # rescaledsaliency = minmax_scale(saliency, axis=1)
    rescaledsaliency = rescaledsaliency.reshape(mask.shape)
    return rescaledsaliency

def getIndexHighest(array, percentageArray):
    X = array.shape[0]
    index = np.argsort(array)
    totalSaliency = np.sum(array)
    indexes = []
    X = 1
    for percentage in percentageArray:
        actualPercentage = percentage / 100
        index_X = index[int(-1 * X):]
        percentageDroped = np.sum(array[index_X]) / totalSaliency
        if (percentageDroped < actualPercentage):
            X = X + 1
            index_X = index[int(-1 * X):]
            percentageDroped = np.sum(array[index_X]) / totalSaliency
            if (not (percentageDroped > actualPercentage)):
                while (percentageDroped < actualPercentage and X < array.shape[0] - 1):
                    X = X + 1
                    index_X = index[int(-1 * X):]
                    percentageDroped = np.sum(array[index_X]) / totalSaliency
        elif (percentageDroped > actualPercentage):
            X = X - 1
            index_X = index[int(-1 * X):]
            percentageDroped_ = np.sum(array[index_X]) / totalSaliency
            if (not (percentageDroped_ < actualPercentage)):

                while (percentageDroped > actualPercentage and X > 1):
                    X = X - 1
                    index_X = index[int(-1 * X):]
                    percentageDroped = np.sum(array[index_X]) / totalSaliency
        indexes.append(index_X)
    return indexes

def save_intoCSV(data,file,Flip=False,col=None,index=False):
	if(Flip):
		print("Will Flip before Saving")
		data=data.reshape((data.shape[1],data.shape[0]))
	df = pd.DataFrame(data)
	# if(col!=None):
	#     df.columns = col
	df.to_csv(file,index=index)

def load_CSV(file,returnDF=False,Flip=False):
	df = pd.read_csv(file)
	data=df.values
	if(Flip):
		print("Will Un-Flip before Loading")
		data=data.reshape((data.shape[1],data.shape[0]))
	if(returnDF):
		return df
	return data

