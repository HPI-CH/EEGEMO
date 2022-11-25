from Crown import Crown
from CrownLive import CrownLive
from Muse import Muse
from MuseLive import MuseLive


class StreamSession:
    """Class to stream eeg data from devices through osc"""

    def get_device(self, args, filename, window_sec=None, model_a=None, model_v=None, logger=None, live=False):
        if live:
            if args['device'] == 'c':
                return CrownLive(args['ip'], 9000, model_a, model_v, window_sec, logger, filename)
            elif args['device'] == 'm':
                return MuseLive(args['ip'], 5000, model_a, model_v, window_sec, logger, filename)
            else:
                raise ValueError('Unrecognized value for device. Device can either be Crown or Muse')
        else:
            if args['device'] == 'c':
                return Crown(args['ip'], 9000, filename)
            elif args['device'] == 'm':
                return Muse(args['ip'], 5000, filename)
            else:
                raise ValueError('Unrecognized value for device. Device can either be Crown or Muse')
