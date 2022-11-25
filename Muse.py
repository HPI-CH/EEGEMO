from datetime import datetime
from Device import Device


class Muse(Device):
    """Class to connect to and read the stream from the Muse S device/ the mind monitor app and write data to a file"""

    # from mindmonitor FAQs about the OSC channels:
    # https://mind-monitor.com/FAQ.php#oscspec
    eeg_channels = ["TP9", "AF7", "AF8", "TP10", "AUX"]
    sampling_frequency = 256

    def __init__(self, ip, port, filename):
        super().__init__("muse", ip, port, filename)
        self.meta_file = self.prepare_meta_file("muse", filename)

    def prepare_meta_file(self, device, filename):
        self.meta_file = open(f"{filename}_{device}_meta.csv", 'w')
        channel_string = ','.join([str(channel) for channel in self.eeg_channels[:-1]])
        self.meta_file.write(f"timeStamp,{channel_string}")
        self.meta_file.write("\n")
        return self.meta_file

    def write_fit_to_file(self, address: str, *args):
        mfile = self.meta_file
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
        mfile.write(timestampStr)
        for arg in args:
            mfile.write(f",{arg}")
        mfile.write("\n")
        mfile.flush()

    def write_eeg_to_file(self, address: str, *args):
        file = self.file
        # mindmonitor does not send a timestamp so it need to be inserted
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
        file.write(timestampStr)
        for arg in args:
            file.write(f",{arg}")
        file.write("\n")
        file.flush()

    def map_streams(self):
        self.dispatcher.map("/muse/eeg", self.write_eeg_to_file)
        self.dispatcher.map("/muse/eeg", self.print_incoming)
        self.dispatcher.map("/muse/elements/horseshoe", self.write_fit_to_file)
