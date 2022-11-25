from pythonosc import dispatcher as disp
from pythonosc import osc_server


class Device:
    """Interface for device classes"""
    eeg_channels = []

    def __init__(self, device_type, ip, port, filename):
        self.ip = ip
        self.port = port
        self.dispatcher = disp.Dispatcher()
        self.server = None
        self.file = self.prepare_file(device_type, filename)
        print(f"{device_type} init done.")

    def prepare_file(self, device, filename):
        self.file = open(f"{filename}_{device}.csv", 'w')
        self.file.write(self.get_header())
        self.file.write("\n")
        return self.file

    def get_header(self):
        channel_string = ','.join([str(channel) for channel in self.eeg_channels])
        header = f"timeStamp,{channel_string},no"
        return header

    def print_incoming(self, address: str, *args):
        print(args)

    # implemented in the classes
    def map_streams(self):
        pass

    def start_stream(self):
        self.map_streams()
        self.server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), self.dispatcher)
        print(f"Serving on {self.server.server_address}")
        self.server.serve_forever()
