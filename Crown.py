from datetime import datetime
from Device import Device


class Crown(Device):
    """Class to connect to and read the stream from the Neurosity Crown device and write data to a file"""

    # from params:
    # ('Crown-000', 'Crown 3', 'Crown', '3', 'Neurosity, Inc', 256, 8, 'CP3,C3,F5,PO3,PO4,F6,C4,CP4')
    eeg_channels = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
    sampling_frequency = 256

    def __init__(self, ip, port, filename):
        super().__init__("crown", ip, port, filename)

    def get_header(self):
        channel_string = ','.join([str(channel) for channel in self.eeg_channels])
        header = f"timeStamp,timeStampCrown,{channel_string},no"
        return header

    def write_to_file(self, address: str, *args):
        file = self.file
        # local timestamp
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
        file.write(f"{timestampStr},")
        # timestamp from device
        file.write(f"{str(args[1])},")
        # measured values
        for arg in args[0]:
            file.write(f"{str(arg)},")
        # counter
        file.write(str(args[2]))
        file.write("\n")
        file.flush()

    def map_streams(self):
        self.dispatcher.map("/neurosity/notion/*", self.write_to_file)
        self.dispatcher.map("/neurosity/notion/*", self.print_incoming)
