import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scikitplot as skplt
from river import metrics as rmetrics
from sklearn.metrics import RocCurveDisplay

import os

logger = logging.getLogger("file_logger")


def plot_periodogram(psd, freqs, ch_names, out):
    for i in range(psd.shape[0]):
        sns.set(font_scale=1.2, style='white')
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd[i], color='k', lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.ylim([0, psd[i].max() * 1.1])
        plt.title(f"Periodogram for channel {ch_names[i]}")
        plt.xlim([0, 45])
        sns.despine()
        plt.savefig(os.path.join(out, f'periodogram_{ch_names[i]}.png'))
        plt.close()


def evaluate_model_plot(y, y_preds, y_probas, dim, out, progressive=True):
    # add a random value (0 or 1) as first element as the prediction is 'none'
    y_probas[0] = {0: np.random.uniform(0, 1), 1: np.random.uniform(0, 1)}

    plot_confusion_matrix(y, y_preds, out)
    plot_roc(y, y_preds, y_probas, out)

    if progressive:
        # plots progressive roc_auc, acc and f1 for one subject/model and returns the values
        return make_progressive_metrics(y, y_preds, y_probas, dim, out)


def plot_roc(y_true, y_pred, y_proba, out):
    fig = RocCurveDisplay.from_predictions(y_true, y_pred)
    fig.plot()
    plt.savefig(os.path.join(out, 'roc1.png'))
    plt.close()
    # fig = skplt.metrics.plot_roc(y_true, y_proba)
    # fig.plot()
    # plt.savefig(os.path.join(out, 'roc2.png'))
    # plt.close()


def plot_confusion_matrix(y_true, y_pred, out, normalized=False):
    fig = skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=normalized)
    fig.plot()
    plt.savefig(os.path.join(out, 'confusion_matrix.png'))
    plt.close()


def plot_tree(model, out, n=1):
    try:
        dot = model.draw()
    except Exception as e:
        logger.info(f"A {e.__class__} occurred in plot_tree.\n")
        return
    dot.filename = f'model_{n}'
    dot.render(directory=out, view=False)


def make_progressive_metrics(y_trues, y_preds, y_probas, dimension, out, step_size=25):
    roc_auc = []
    acc = []
    f1 = []
    example_sizes = []
    metric_roc_auc = rmetrics.ROCAUC()
    metric_acc = rmetrics.Accuracy()
    metric_f1 = rmetrics.F1()

    for index, y_true in enumerate(y_trues):
        metric_roc_auc = metric_roc_auc.update(y_true, y_probas[index])
        metric_acc = metric_acc.update(y_true, y_preds[index])
        metric_f1 = metric_f1.update(y_true, y_preds[index])
        if index % step_size == 0:
            example_sizes.append(index + 1)
            roc_auc.append(metric_roc_auc.get())
            acc.append(metric_acc.get())
            f1.append(metric_f1.get())

    plot_progressive_metrics_single(example_sizes, roc_auc, acc, f1, dimension, out)

    rates_string = f"Progressive Metrics:\n" \
                   f"ROCAUC: {roc_auc[-1:]}\n" \
                   f"Accuracy: {acc[-1:]}\n" \
                   f"F1-Score: {f1[-1:]}\n"

    logger.info(rates_string)

    return roc_auc, acc, f1, example_sizes


def plot_progressive_metrics_single(example_sizes, roc_auc, acc, f1, dimension, out):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(f"Progressive Metrics: {dimension}")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("ROCAUC")
    axes[0].grid()
    axes[0].plot(example_sizes, roc_auc, "o-", color="b")

    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid()
    axes[1].plot(example_sizes, acc, "o-", color="b")

    axes[2].set_xlabel("Training examples")
    axes[2].set_ylabel("F1 Score")
    axes[2].grid()
    axes[2].plot(example_sizes, f1, "o-", color="b")

    plt.savefig(os.path.join(out, 'progressive_metrics.png'))
    plt.close()


def plot_progressive_metrics_all(example_sizes, roc_auc, acc, f1, dimension, logger, out):
    # get the shortest list
    example_size = min(example_sizes, key=len)
    min_size = len(example_size)

    # shorten the lists
    roc_auc_lim = [ll[:min_size] for ll in roc_auc]
    acc_lim = [ll[:min_size] for ll in acc]
    f1_lim = [ll[:min_size] for ll in f1]

    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    roc_auc_mean = np.mean(roc_auc_lim, axis=0)
    roc_auc_std = np.std(roc_auc_lim, axis=0)
    acc_mean = np.mean(acc_lim, axis=0)
    acc_std = np.std(acc_lim, axis=0)
    f1_mean = np.mean(f1_lim, axis=0)
    f1_std = np.std(f1_lim, axis=0)

    logger.info(f"Overall Metrics Means {dimension}:\n")
    metrics_string = f"# Samples: {example_size}\n" \
                     f"ROC AUC Mean: {roc_auc_mean}\n" \
                     f"Accuracy Mean: {acc_mean}\n" \
                     f"F1 Score Mean: {f1_mean}\n"

    logger.info(metrics_string)

    axes[0].set_title(f"Progressive Metrics: {dimension}")
    roc_auc_min = roc_auc_mean - roc_auc_std
    roc_auc_min[roc_auc_min < 0] = 0
    roc_auc_max = roc_auc_mean + roc_auc_std
    roc_auc_max[roc_auc_max > 1] = 1
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("ROCAUC")
    axes[0].grid()
    axes[0].fill_between(
        example_size,
        roc_auc_min,
        roc_auc_max,
        alpha=0.1,
    )
    axes[0].plot(example_size, roc_auc_mean, "o-", color="b")

    acc_min = acc_mean - acc_std
    acc_min[acc_min < 0] = 0
    acc_max = acc_mean + acc_std
    acc_max[acc_max > 1] = 1
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid()
    axes[1].fill_between(
        example_size,
        acc_min,
        acc_max,
        alpha=0.1,
    )
    axes[1].plot(example_size, acc_mean, "o-", color="b")

    f1_min = f1_mean - f1_std
    f1_min[f1_min < 0] = 0
    f1_max = f1_mean + f1_std
    f1_max[f1_max > 1] = 1
    axes[2].set_xlabel("Training examples")
    axes[2].set_ylabel("F1 Score")
    axes[2].grid()
    axes[2].fill_between(
        example_size,
        f1_min,
        f1_max,
        alpha=0.1,
    )
    axes[2].plot(example_size, f1_mean, "o-", color="b")

    plt.savefig(os.path.join(out, 'progressive_metrics.png'))
    plt.close()


def plot_learning_curve(train_sizes, train_scores, validation_scores, metric, out):
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label=f'Training {metric}')
    plt.plot(train_sizes, validation_scores_mean, label=f'Validation {metric}')
    plt.ylabel(f'{metric}', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.savefig(os.path.join(out, f'learning_curve_{metric}.png'))
    plt.close()
