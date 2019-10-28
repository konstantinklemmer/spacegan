import torch
import pandas as pd
import numpy as np
import pysal.lib
import spacegan_method


def get_spacegan_config(training_step, prob_config, check_config, cond_input, target):

    # load generator
    gen_model = torch.load("gen_iter " + str(training_step) + ".pkl.gz")

    # load discrimininator
    disc_model = torch.load("disc_iter " + str(training_step) + ".pkl.gz")

    # create a cgan object
    spacegan_i = spacegan_method.SpaceGAN(prob_config, check_config, disc_model, gen_model)

    # get scaling method
    spacegan_i.fit_scaling_method(cond_input, target)

    return spacegan_i


def compute_metrics(target, cond_input, prob_config, check_config, coord_input, neighbours):

    # pre-allocation
    perf_metrics_dict = check_config["perf_metrics"]
    agg_funcs_dict = check_config["agg_funcs"]
    check_interval = check_config["check_interval"]
    epochs = prob_config["epochs"]
    n_samples = check_config["n_samples"]
    spacegan_iters = [x * check_interval for x in range(0, int(epochs / check_interval))] + [epochs]
    spacegan_dict = {}

    for i in spacegan_iters:
        spacegan_dict[str(i)] = get_spacegan_config(i, prob_config, check_config, cond_input, target)
        print("Loaded checkpoint %d" % i, end="\r")

    # compute metrics
    # pre-allocation
    df_pred_metrics, df_agg_pred, = {}, {}
    df_pred_agg_metrics = pd.DataFrame(index=list(spacegan_dict.keys()), columns=list(perf_metrics_dict.keys()))
    for pf in list(perf_metrics_dict.keys()):
        df_pred_metrics[pf] = pd.DataFrame(index=list(range(n_samples)), columns=list(spacegan_dict.keys()))

        if pf in ["MIE", "MIEPs"]:
            # distance matrix and normalization
            dist = pysal.lib.cg.distance_matrix(coord_input)
            u_dist = np.unique(dist)
            k_min_dist = np.sort(u_dist.flatten())[:neighbours]
            kd = pysal.lib.cg.kdtree.KDTree(coord_input)
            wdist = pysal.lib.weights.distance.DistanceBand(kd, threshold=k_min_dist[2], binary=True, p=2)  # Queen
            wdist.transform = "r"

    for agg in list(agg_funcs_dict.keys()):
        df_agg_pred[agg] = pd.DataFrame(columns=list(spacegan_dict.keys()))

    # main loop
    for i in list(spacegan_dict.keys()):
        spacegan_predictions = np.array([])

        # sampling and performance
        for n in range(n_samples):
            if n is 0:
                spacegan_predictions = spacegan_dict[i].predict(cond_input)
            else:
                spacegan_predictions = np.concatenate([spacegan_predictions, spacegan_dict[i].predict(cond_input)], axis=1)

            # compute performance
            if check_config["sample_metrics"]:
                for pf in list(perf_metrics_dict.keys()):
                    if pf in ["MIE", "MIEPs"]:
                        df_pred_metrics[pf].loc[n, i] = perf_metrics_dict[pf](spacegan_predictions[:, n], target, wdist)
                    else:
                        df_pred_metrics[pf].loc[n, i] = perf_metrics_dict[pf](spacegan_predictions[:, n], target)

        # aggregation
        for agg in list(agg_funcs_dict.keys()):
            df_agg_pred[agg][i] = agg_funcs_dict[agg](spacegan_predictions, axis=1)
        print("Sampled from checkpoint " + i, end="\r")

    # performance metrics for aggregated
    if check_config["agg_metrics"]:
        for pf in list(perf_metrics_dict.keys()):
            print("Evaluating using metric " + pf)
            for i in list(spacegan_dict.keys()):
                if pf in ["MIE", "MIEPs"]:
                    df_pred_agg_metrics.loc[i, pf] = perf_metrics_dict[pf](df_agg_pred["avg"][[i]].values, target, wdist)
                else:
                    df_pred_agg_metrics.loc[i, pf] = perf_metrics_dict[pf](df_agg_pred["avg"][[i]].values, target)
                print("Checkpoint " + i, end="\r")

    # export results
    return {"sample_metrics": df_pred_metrics, "agg_metrics": df_pred_agg_metrics}
