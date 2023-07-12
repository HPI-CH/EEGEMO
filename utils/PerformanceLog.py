import os


class PerformanceLog:
    COLUMN_NAMES = ['Model', 'N', 'Trial_Count', 'Evaluation_Procedure', 'Features', 'Arousalrate',
                    'Dataset', 'Device', 'Window_size', 'Dimension', 'Subject', '#Datapoints', 'prog_ROCAUC',
                    'prog_Accuracy', 'prog_F1', 'cv_ROCAUC', 'cv_Accuracy', 'cv_F1', 'Hyper_Parameters']

    def __init__(self, trial_count, dataset, device, features):
        self.sampling_rate = None
        self.n = None
        self.hyper_param = None
        self.model = None
        self.trial_count = trial_count
        self.dataset = dataset
        self.device = device
        self.features = features
        self.file = self.prepare_file('performance_comparisons')

    def set_model_params(self, model, n, sampling_rate, **kwargs):
        self.model = model
        self.n = n
        self.hyper_param = kwargs
        self.sampling_rate = sampling_rate

    def prepare_file(self, filename):
        file = f"{filename}.csv"
        with open(file, 'a') as f:
            f.write("\n")
            if os.stat(file).st_size == 0:
                f.write(self.get_header())
                f.write("\n")
        return file

    def get_header(self):
        header = ','.join([entry for entry in self.COLUMN_NAMES])
        return header

    def write_line(self, f, roc_auc, acc, f1, subject, size, dimension, window_size, cv_roc_auc, cv_acc, cv_f1):
        values = [self.model, self.n, self.trial_count, '', self.features, self.sampling_rate, self.dataset,
                  self.device, window_size, dimension, subject, size, roc_auc, acc, f1, cv_roc_auc, cv_acc, cv_f1]
        values_joined = ','.join([str(element) for element in values])
        hyper_params = ','.join([f'{k}: {v}' for k, v in self.hyper_param.items()])
        line = f'{values_joined},{hyper_params}'
        f.write(line)
        f.write("\n")

    def write_lines(self, roc_auc, acc, f1, subject, example_sizes, dimension, window_size, cv_roc_auc, cv_acc, cv_f1):
        with open(self.file, 'a') as f:
            for ii in range(len(roc_auc)):
                self.write_line(f, roc_auc[ii], acc[ii], f1[ii], subject, example_sizes[ii], dimension, window_size,
                                cv_roc_auc, cv_acc, cv_f1)
