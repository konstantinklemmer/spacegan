import pandas as pd
import numpy as np
import esda


def gaussian(size=(1, 1), **kwargs):
    if kwargs["params"] is None:
        return np.random.normal(size=size)
    else:
        noise = np.random.normal(size=size)
        if "scale" in kwargs["params"].keys():
            noise *= kwargs["params"]["scale"]
        if "loc" in kwargs["params"].keys():
            noise += kwargs["params"]["loc"]
        return noise


def ts_df(ts, lags):
    df = pd.DataFrame(index=ts.index, columns=["lag_" + str(x) for x in range(lags+1)])
    for i in range(lags+1):
        df["lag_" + str(i)] = ts.shift(i)

    return df.dropna()


def rmse(obs, pred):
    return np.sqrt(np.mean(np.square(obs-pred)))


def mad(obs, pred):
    return np.mean(np.abs(obs-pred))


def pearsoncorr(obs, pred):
    return np.corrcoef(obs, pred, rowvar=False)[0, 1]


def mie(obs, pred, wdist):
    # compute Moran's Is
    np.random.seed(12345)
    local_mi = esda.moran.Moran_Local(obs, wdist, permutations=100)
    local_mi_target = local_mi.Is

    local_mi = esda.moran.Moran_Local(pred, wdist, permutations=100)
    local_mi_pred = local_mi.Is

    return np.sqrt(np.mean((local_mi_target-local_mi_pred) ** 2.0))


def moranps(obs, pred, wdist):
    # compute Moran's Is
    np.random.seed(12345)
    local_mi = esda.moran.Moran_Local(obs, wdist, permutations=500)
    local_mi_target = local_mi.p_sim

    local_mi = esda.moran.Moran_Local(pred, wdist, permutations=500)
    local_mi_pred = local_mi.p_sim

    return np.sqrt(np.mean((local_mi_target-local_mi_pred) ** 2.0))


def mase_1(obs, pred):
    return mad(obs, pred)/np.mean(np.abs(np.diff(obs, axis=0)))


def mape(obs, pred):
    return np.mean(np.abs(obs-pred)/np.abs(obs))


def smape(obs, pred):
    return np.mean(np.abs(obs - pred) / (np.abs(obs + pred) / 2.0))


def eool(obs, pred, perc):
    res = np.abs(obs - pred)
    return np.mean(res[res > np.percentile(res, perc)])


def msis_1(obs, upper, lower, alpha):
    den = np.mean(np.abs(np.diff(obs, axis=0)))
    width = np.mean(upper-lower)
    low_cov = (2.0/alpha) * np.mean((lower-obs) * (obs < lower))
    up_cov = (2.0 / alpha) * np.mean((obs - upper) * (obs > upper))
    return (width + low_cov + up_cov)/den


def get_neighbours_featurize(sp_df, spatial_coords, features, nn_size):
    # find neighbors
    from sklearn.neighbors import NearestNeighbors
    nneighbors_method = NearestNeighbors(n_neighbors=1 + nn_size).fit(sp_df[spatial_coords].values)
    nneighbors = nneighbors_method.kneighbors(X=sp_df[spatial_coords], return_distance=False)

    # labels for additional features
    label_list = []
    for out in features:
        label_list += ["nn_" + out + "_" + str(i) for i in range(nn_size)]
    aug_sp_df = pd.DataFrame(columns=label_list, index=sp_df.index)

    # get features and featurize them
    for i in sp_df.index:
        aug_sp_df.loc[i] = sp_df.iloc[nneighbors[i][1:]][features].values.reshape(-1, 1, order="F").transpose()

    return pd.concat([sp_df, aug_sp_df], axis=1), label_list
