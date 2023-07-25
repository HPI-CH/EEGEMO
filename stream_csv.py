import argparse
import itertools
import joblib
import logging
import logging.config
import os
import pandas as pd
import numpy as np
from numpy.random import randint
from river import compose
from river import compat
from river import ensemble
from river import imblearn
from river import preprocessing
from river import stream
from river import tree
from sklearn import metrics, model_selection
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score

from Amigos import Amigos
from FeatureWindow import FeatureWindow
from MyData import MyData
from utils.PerformanceLog import PerformanceLog
from utils import plots

logger = logging.getLogger("file_logger")
window_seconds = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Connects to OSC-streams from portable EEG-devices and saves the incoming data to csv files."
    )
    parser.add_argument(
        "input_dir", type=str, help="Input directory including path")
    parser.add_argument("dataset", type=str, help="A(migos) or M(y own data)", choices=['A', 'M'])
    parser.add_argument("subjectID", type=int, help="Subject ID")
    parser.add_argument("device", type=str, help="Device used for collecting the data. Either m(use) or c(rown)",
                        choices=['m', 'c'])
    parser.add_argument("--trialCount", type=int, default=1,
                        help="The amount of times the trials should be run. Default is 1")
    parser.add_argument("--oversamplingRate", type=int,default = 30,
                        help="The percentage rate at which the 0 category for Arousal should be oversampled."
                             "Default is 30")
    parser.add_argument("--window_seconds", type=int, default=3,
                        help="The size of the feature window in seconds")
    parser.add_argument("--performanceComparison", type=bool, default=False,
                        help="")
    parser.add_argument(
        "--out", type=str, help="Output directory. Default ./output_data_amigos/"
    )

    return vars(parser.parse_args())


def train_predict(X, y, model, window, delay=0, labels_package=True):
    x_extracted = []
    y_true = []
    y_preds = []
    y_probas = []
    delay_left = delay
    idx = -1

    for xi, yi in stream.iter_pandas(X, y):
        features = window.next(xi, yi)
        if features:
            idx += 1
            # Obtain the prior prediction and update the model in one go
            x_extracted.append(window.x)
            y_true.append(window.y)
            pred = model.predict_one(window.x)
            if pred is None:
                y_preds.append(randint(2))
            else:
                y_preds.append(model.predict_one(window.x))
            y_probas.append(model.predict_proba_one(window.x))
            if delay_left == 0:
                if labels_package:
                    # all the labels since last seen labels arrive together after a certain delay
                    # -> like in an experiment
                    for ii in range(delay + 1):
                        model.learn_one(x_extracted[idx - ii], y_true[idx - ii])
                else:
                    # one label arrives after a certain delay
                    model.learn_one(x_extracted[idx - delay], y_true[idx - delay])
                delay_left = delay
            else:
                delay_left -= 1

    return x_extracted, y_true, y_preds, y_probas, model


def compute_rates(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except Exception as e:
        logger.info(f"A {e.__class__} occurred in compute_rates.\n There is most likely just one true class.")
        return

    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (tp + fn)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    rates_string = f"Metrics:\n" \
                   f"False positive rate: {false_positive_rate}\n" \
                   f"False negative rate: {false_negative_rate}\n" \
                   f"Recall/Sensitivity: {recall}\n" \
                   f"Precision: {precision}\n" \
                   f"Accuracy: {accuracy}\n" \
                   f"F1-Score: {f1}\n"

    logger.info(rates_string)


def cross_validate(X, y, model, train_sizes, n_splits=5, plot=False):
    x_list = [list(d.values()) for d in X]

    # Define a deterministic cross-validation procedure
    cv = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Compute the MSE values
    scorer = metrics.make_scorer(metrics.roc_auc_score)
    scorer_f1 = metrics.make_scorer(metrics.f1_score)
    scorer_acc = metrics.make_scorer(metrics.accuracy_score)

    # We make the Pipeline compatible with sklearn
    model = compat.convert_river_to_sklearn(model)

    # We compute the CV scores using the same CV scheme and the same scoring
    scores_roc = model_selection.cross_val_score(model, x_list, y, scoring=scorer, cv=cv)
    scores_f1 = model_selection.cross_val_score(model, x_list, y, scoring=scorer_f1, cv=cv)
    scores_acc = model_selection.cross_val_score(model, x_list, y, scoring=scorer_acc, cv=cv)

    # Display the average score and it's standard deviation
    logger.info("CV overall Metrics:")
    metrics_string = f'ROC AUC: {scores_roc.mean():.3f} (± {scores_roc.std():.3f})\n' \
                     f'F1: {scores_f1.mean():.3f} (± {scores_f1.std():.3f})\n' \
                     f'Accuracy: {scores_acc.mean():.3f} (± {scores_acc.std():.3f})'

    logger.info(metrics_string)

    if plot:
        train_sizes, train_scores_roc, validation_scores_roc = model_selection.learning_curve(estimator=model, X=x_list,
                                                                                              y=y,
                                                                                              train_sizes=train_sizes,
                                                                                              cv=n_splits,
                                                                                              scoring=scorer)
        plots.plot_learning_curve(train_sizes, train_scores_roc, validation_scores_roc)
        train_sizes, train_scores_acc, validation_scores_acc = model_selection.learning_curve(estimator=model,
                                                                                              X=x_list,
                                                                                              y=y,
                                                                                              train_sizes=train_sizes,
                                                                                              cv=n_splits,
                                                                                              scoring=scorer_acc)
        plots.plot_learning_curve(train_sizes, train_scores_acc, validation_scores_acc)
        train_sizes, train_scores_f1, validation_scores_f1 = model_selection.learning_curve(estimator=model,
                                                                                            X=x_list,
                                                                                            y=y,
                                                                                            train_sizes=train_sizes,
                                                                                            cv=n_splits,
                                                                                            scoring=scorer_f1)
        plots.plot_learning_curve(train_sizes, train_scores_f1, validation_scores_f1)

    return scores_roc.mean(), scores_acc.mean(), scores_f1.mean()


def build_model(dimension, arousal_rate=None, performance_log=None):
    n_models = 3
    grace_period=30
    seed = 42
    # dataset = stream.iter_pandas(X, y)
    base_model = tree.HoeffdingAdaptiveTreeClassifier(grace_period=grace_period)  # , drift_window_threshold=50)
    if dimension == 'Arousal' and arousal_rate:
        low_arousal_rate = arousal_rate / 10
        high_arousal_rate = 1 - low_arousal_rate
        model = imblearn.RandomOverSampler(
            (
                    preprocessing.StandardScaler() |
                    ensemble.SRPClassifier(model=base_model, n_models=n_models)
            ),
            desired_dist={False: low_arousal_rate, True: high_arousal_rate}
        )
    else:
        model = compose.Pipeline(
            preprocessing.StandardScaler(),
            # tree.HoeffdingAdaptiveTreeClassifier(grace_period=50)
            # tree.HoeffdingAdaptiveTreeClassifier(grace_period=10, seed=seed))
            ensemble.SRPClassifier(model=base_model, n_models=n_models))
        # ensemble.AdaptiveRandomForestClassifier(n_models=3, seed=42, grace_period=30))

    logger.info(model)
    logger.info(f"n_models: {n_models}")
    if performance_log:
        performance_log.set_model_params(model='RandomOverSampler StandardScaler SRPClassifier HoeffdingAdaptiveTreeClassifier', n=n_models, sampling_rate=low_arousal_rate,
                                         grace_period=grace_period)

    return model


def debug_model(X, Y, out, model=tree.HoeffdingAdaptiveTreeClassifier(grace_period=50)):
    dataset = stream.iter_pandas(X, Y)
    for x, y in itertools.islice(dataset, 100000):
        model.predict_one(x)
        model.learn_one(x, y)

    x, y = next(dataset)
    print(model.debug_one(x))
    dot = model.draw()
    dot.render(directory=out, view=False)


def shuffle_data(df_eeg, df_labels, windows=True, size=1):
    to_shuffle = pd.concat([df_eeg, df_labels], axis=1)
    if windows:
        # shuffle windows (so that the data inside the windows remains unshuffled
        # and the data inside one window always belongs to the same video)
        np.random.shuffle(to_shuffle.values.reshape(-1, size, to_shuffle.shape[1]))
        shuffled = to_shuffle
    else:
        shuffled = shuffle(to_shuffle)
    shuffled = shuffled.reset_index(drop=True)
    labels_shuffled = shuffled[['Valence', 'Arousal']]
    eeg_shuffled = shuffled.drop(['Valence', 'Arousal'], axis=1)
    return eeg_shuffled, labels_shuffled


def make_out_path(out, dimension):
    out_path = os.path.join(out, dimension)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    return out_path


def prepare_single_subject(dataset, user_id, ws):
    logger.info(f"Subject: {user_id}\n")
    df_eeg, df_labels = dataset.prepare_data(user_id, ws)
    return df_eeg, df_labels


def run_single_subject(dataset, ws, arousal_rate, out, df_eeg, df_labels, keep_model=False, models=None, debug=False,
                       save=False,
                       plot=True):
    # shuffle data
    df_eeg_shuffled, df_labels_shuffled = shuffle_data(df_eeg, df_labels, windows=True,
                                                       size=dataset.sampling_frequency * window_seconds)
    metrics = {}
    for dimension in ['Valence', 'Arousal']:
        if keep_model:
            model = models[dimension]
        else:
            model = build_model(dimension, arousal_rate)
        logger.info(dimension)
        out_path = make_out_path(out, dimension)
        window = FeatureWindow(ws, dataset.sampling_frequency, dataset.eeg_channels, live=False)
        x_extracted, y_true, y_preds, y_probas, model_trained = train_predict(df_eeg_shuffled,
                                                                              df_labels_shuffled[dimension],
                                                                              model, window)

        # compare(x_extracted, y_true, model)
        if plot:
            roc_auc, acc, f1, example_sizes = plots.evaluate_model_plot(y_true, y_preds, y_probas, dimension, out_path)

            compute_rates(y_true, y_preds)
            metrics[dimension] = (y_true, y_preds, y_probas, roc_auc, acc, f1, example_sizes)
            # for n, model in enumerate(model_trained["AdaptiveRandomForestClassifier"].models):
            try:
                for n, model in enumerate(model_trained["SRPClassifier"].models):
                    out_path = os.path.join(out_path)
                    plots.plot_tree(model.model, out_path, n)
            except Exception:
                for n, model in enumerate(model_trained.classifier["SRPClassifier"].models):
                    out_path = os.path.join(out_path)
                    plots.plot_tree(model.model, out_path, n)
        else:
            metrics[dimension] = (y_true, y_preds, y_probas)

        if debug:
            debug_model(x_extracted, y_true, out_path)

        if save:
            model_file = os.path.join(out_path, 'model.sav')
            joblib.dump(model_trained, model_file)

    return metrics


def run_all(dataset, arousal_rate, trial_count, ws, out, keep_model=False, debug=False, save=False, plot=True):
    metrics_all = []
    models = None
    if keep_model:
        model_a = build_model('Arousal', arousal_rate)
        model_v = build_model('Valence')
        models = {"Valence": model_v, "Arousal": model_a}

    if dataset.name == "My Dataset":
        #rrange = range(3, 14)
        rrange = range(3, 5)
    else:
        rrange = range(1, dataset.length + 1)

    for subject in rrange:
        # the Crown data for subject 5 is way shorter than for all the others because of connectivity issues
        # if dataset.name == "My Dataset" and dataset.device == 'crown' and subject == 5:
        #     print(subject)
        #     continue
        metrics_subj = []
        out_subj = os.path.join(out, str(subject))
        df_eeg, df_labels = prepare_single_subject(dataset, subject, ws)

        for n in range(trial_count):
            metrics_subj.append(
                run_single_subject(dataset, ws, arousal_rate, out_subj, df_eeg, df_labels, keep_model, models, debug, save))

        metrics_all.append(metrics_subj)
        if plot:
            # plot metrics per subjects
            make_eval_all(metrics_subj, out_subj)

    return metrics_all


def make_eval_all(metrics_all, out):
    for dimension in ['Valence', 'Arousal']:
        flat_y_trues = [item for dict in metrics_all for item in dict[dimension][0]]
        # flat_y_trues = [item for subj in metrics_all for dict in subj for item in dict[dimension][0]]
        flat_y_preds = [item for dict in metrics_all for item in dict[dimension][1]]
        flat_y_probas = [item for dict in metrics_all for item in dict[dimension][2]]

        roc_auc_all = [dict[dimension][3] for dict in metrics_all]
        acc_all = [dict[dimension][4] for dict in metrics_all]
        f1_all = [dict[dimension][5] for dict in metrics_all]
        example_sizes_all = [dict[dimension][6] for dict in metrics_all]

        out_path = make_out_path(out, dimension)
        plots.plot_confusion_matrix(flat_y_trues, flat_y_preds, out_path)
        plots.plot_progressive_metrics_all(example_sizes_all, roc_auc_all, acc_all, f1_all, dimension, logger,
                                           out_path)
        logger.info(f"Overall Metrics {dimension}:\n")
        compute_rates(flat_y_trues, flat_y_preds)


def load_model(model_file):
    return joblib.load(model_file)


def get_dataset(set, dirpath, device):
    if set == 'A':
        return Amigos(dirpath)
    elif set == 'M':
        return MyData(dirpath, device)
    else:
        raise ValueError('Unrecognized value for dataset. Dataset can either be Amigos or My own data')


def main():
    #logging.config.fileConfig("./logs.config")

    args = parse_args()

    set_name = args["dataset"]
    subject_id = args["subjectID"]
    device = args["device"]
    dirname = args["input_dir"]
    arousal_rate = args["oversamplingRate"]
    trial_count = args["trialCount"]
    window_seconds = args["window_seconds"]
    performanceComp = args["performanceComparison"]

    dataset = get_dataset(set_name, dirname, device)
    logger.info(f"Dataset: {dataset.name}, Window Size: {window_seconds} seconds, "
                f"Low Arousal sampling rate: {arousal_rate}, Trial Count: {trial_count}")

    if not args["out"]:
        out_path = dataset.out_path
    else:
        out_path = args["out"]
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    logger.info(f"Belongs to: {out_path}")

    metrics_overall = []

    if performanceComp:
        performance_log = PerformanceLog(trial_count, dataset.name, device, features='Bandpower BandPowerRatio')
        if dataset.name == "My Dataset":
            rrange = range(3, 14)
        else:
            rrange = range(1, dataset.length + 1)

        metrics_dataset = {'Arousal': [], 'Valence': []}
        for subject in rrange:
            metrics_subj = []
            df_eeg, df_labels = prepare_single_subject(dataset, subject, window_seconds)

            for dimension in ['Valence', 'Arousal']:
                for n in range(trial_count):
                    # shuffle data
                    df_eeg_shuffled, df_labels_shuffled = shuffle_data(df_eeg, df_labels, windows=True,
                                                                       size=dataset.sampling_frequency * window_seconds)
                    model = build_model(dimension, arousal_rate)
                    logger.info(dimension)
                    out_path = make_out_path(out_path, dimension)
                    window = FeatureWindow(window_seconds, dataset.sampling_frequency, dataset.eeg_channels, live=False)
                    x_extracted, y_true, y_preds, y_probas, model_trained = train_predict(df_eeg_shuffled,
                                                                                          df_labels_shuffled[dimension],
                                                                                          model, window)

                    roc_auc, acc, f1, example_sizes = plots.make_progressive_metrics(y_true, y_preds, y_probas,
                                                                                     dimension, out_path)
                    metrics_subj.append([roc_auc, acc, f1])

                roc_auc_subj = [item[0] for item in metrics_subj]
                acc_subj = [item[1] for item in metrics_subj]
                f1_subj = [item[2] for item in metrics_subj]
                roc_auc_mean = np.nanmean(roc_auc_subj, axis=0)
                acc_mean = np.nanmean(acc_subj, axis=0)
                f1_mean = np.nanmean(f1_subj, axis=0)

                model = build_model(dimension, arousal_rate, performance_log)
                cv_roc, cv_acc, cv_f1 = cross_validate(x_extracted, y_true, model, example_sizes)
                performance_log.write_lines(roc_auc_mean, acc_mean, f1_mean, subject, example_sizes, dimension,
                                            window_seconds, cv_roc, cv_acc, cv_f1)

                metrics_dataset[dimension].append([roc_auc_mean, acc_mean, f1_mean, cv_roc, cv_acc, cv_f1])
        # write metrics per dim for whole dataset -> means
        for dim in ['Valence', 'Arousal']:
            roc_auc_all = [item[0] for item in metrics_dataset[dim]]
            # get the shortest list
            example_size = min(roc_auc_all, key=len)
            min_size = len(example_size)
            acc_all = [item[1] for item in metrics_dataset[dim]]
            f1_all = [item[2] for item in metrics_dataset[dim]]
            cv_roc_all = [item[3] for item in metrics_dataset[dim]]
            cv_acc_all = [item[4] for item in metrics_dataset[dim]]
            cv_f1_all = [item[5] for item in metrics_dataset[dim]]

            # shorten the lists
            roc_auc_lim = [ll[:min_size] for ll in roc_auc_all]
            acc_lim = [ll[:min_size] for ll in acc_all]
            f1_lim = [ll[:min_size] for ll in f1_all]

            roc_auc_all_mean = np.nanmean(roc_auc_lim, axis=0)
            acc_all_mean = np.nanmean(acc_lim, axis=0)
            f1_all_mean = np.nanmean(f1_lim, axis=0)
            cv_roc_mean = np.nanmean(cv_roc_all)
            cv_acc_mean = np.nanmean(cv_acc_all)
            cv_f1_mean = np.nanmean(cv_f1_all)
            performance_log.write_lines(roc_auc_all_mean, acc_all_mean, f1_all_mean, 'all', example_sizes, dim,
                                        window_seconds, cv_roc_mean, cv_acc_mean, cv_f1_mean)

    else:
        if subject_id:
            out_path = os.path.join(out_path, str(subject_id))
            print(out_path)
            df_eeg, df_labels = prepare_single_subject(dataset, subject_id, window_seconds)
            for n in range(trial_count):
                metrics_overall.append(
                    run_single_subject(dataset, window_seconds, arousal_rate, out_path, df_eeg, df_labels, save=True))
        else:
            metrics_combined = run_all(dataset, arousal_rate, trial_count, window_seconds, out_path, keep_model=True,save=True)
            metrics_overall = [evals for subj in metrics_combined for evals in subj]

        # plot overall metric
        make_eval_all(metrics_overall, out_path)

    logging.shutdown()


if __name__ == '__main__':
    main()
