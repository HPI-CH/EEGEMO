from datetime import datetime
from Device import Device
import joblib
import os


class DeviceLive(Device):
    """Class to connect to and read the stream from the Neurosity Crown device and do live predictions on the data"""

    def __init__(self, ip, port, device, model_a, model_v, window_sec, logger, filename):
        Device.__init__(self, device, ip, port, filename)
        self.logger = logger
        self.model_a = model_a
        self.model_v = model_v
        self.window_sec = window_sec
        self.predictions_a = {}
        self.predictions_v = {}
        self.probas_a = {}
        self.probas_v = {}
        self.extracted_x = {}
        self.window = None

    def predict_incoming(self, *values):
        eeg_data = []
        # local timestamp
        datetime_now = datetime.now()
        # measured values
        for val in values[0]:
            eeg_data.append(val)

        ready = self.window.next(eeg_data, None)
        if ready:
            # to make sure the data is still coming in properly
            self.logger.info(f"Data at {datetime_now}: {eeg_data}")

            # Arousal
            pred_a = self.model_a.predict_one(self.window.x)
            proba_a = self.model_a.predict_proba_one(self.window.x)
            self.logger.info(f"Prediction for Arousal at {datetime_now}: {proba_a}, i.e. {pred_a}")
            # Valence
            pred_v = self.model_v.predict_one(self.window.x)
            proba_v = self.model_v.predict_proba_one(self.window.x)
            self.logger.info(f"Prediction for Valence at {datetime_now}: {proba_v}, i.e. {pred_v}")

            # store values for later
            self.extracted_x[datetime_now] = self.window.x
            self.predictions_a[datetime_now] = pred_a
            self.probas_a[datetime_now] = proba_a
            self.predictions_v[datetime_now] = pred_v
            self.probas_v[datetime_now] = proba_v

    def learn_from_labels(self, address: str, *args):
        vid_start = args[1]
        vid_end = args[2]
        label = 1 if args[3] >= 0.5 else 0

        # get all the data points that lie in the video time period for which there is a label
        for ts in self.extracted_x.keys():
            if float(vid_start) <= ts.timestamp() <= float(vid_end):
                try:
                    if "A" in args[0]:
                        self.model_a.learn_one(x=self.extracted_x[ts], y=int(label))
                    else:
                        self.model_v.learn_one(x=self.extracted_x[ts], y=int(label))
                except Exception as e:
                    self.logger.info(f"A {e.__class__} occurred in learn_from_labels:\n {e}")
                    continue

        self.logger.info(f"Video start, video end, dimension, label, label_norm\n"
                         f" {vid_start}, {vid_end}, {args[0]}, {args[3]}, {label}")

        # save/override models after learning
        self.save_models()

    def save_models(self):
        a_file = os.path.join('output_data', 'model_arousal_after.sav')
        v_file = os.path.join('output_data', 'model_valence_after.sav')
        joblib.dump(self.model_a, a_file)
        joblib.dump(self.model_v, v_file)
