from Muse import Muse
from DeviceLive import DeviceLive
from FeatureWindow import FeatureWindow


class MuseLive(DeviceLive, Muse):
    """Class to connect to and read the stream from the Muse S device/ the mind monitor app
    and do live predictions on the data """

    def __init__(self, ip,subjID,sessionName, port, model_a, model_v, window_sec, logger, filename):
        super().__init__(ip,subjID,sessionName, port, "muse", model_a, model_v, window_sec, logger, filename)
        self.window = FeatureWindow(self.window_sec, self.sampling_frequency, self.eeg_channels, live=True)

    def send_to_predict(self, address: str, *args):
        values = args
        super().predict_incoming(values)

    def map_streams(self):
        self.dispatcher.map("/muse/eeg", self.write_eeg_to_file)
        self.dispatcher.map("/muse/eeg", self.send_to_predict)
        self.dispatcher.map("/psychopy/valence", super().learn_from_labels, "V")
        self.dispatcher.map("/psychopy/arousal", super().learn_from_labels, "A")
     