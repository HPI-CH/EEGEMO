from Crown import Crown
from DeviceLive import DeviceLive
from FeatureWindow import FeatureWindow


class CrownLive(DeviceLive, Crown):
    """Class to connect to and read the stream from the Neurosity Crown device and do live predictions on the data"""

    def __init__(self, ip, port, model_a, model_v, window_sec, logger, filename):
        super().__init__(ip, port, "crown", model_a, model_v, window_sec, logger, filename)
        self.window = FeatureWindow(self.window_sec, self.sampling_frequency, self.eeg_channels, live=True)

    def send_to_predict(self, address: str, *args):
        values = args[0]
        # every now and then the Crown sends a weird signature that messes everything up
        if len(values) == len(self.eeg_channels):
            super().predict_incoming(values)

    def map_streams(self):
        self.dispatcher.map("/neurosity/notion/*", self.write_to_file)
        self.dispatcher.map("/neurosity/notion/*", self.send_to_predict)
        self.dispatcher.map("/psychopy/valence", super().learn_from_labels, "V")
        self.dispatcher.map("/psychopy/arousal", super().learn_from_labels, "A")
