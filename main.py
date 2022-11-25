import argparse
import os

from StreamSession import StreamSession


def parse_args():
    parser = argparse.ArgumentParser(
        description="Connects to OSC-streams from portable EEG-devices and saves the incoming data to csv files."
    )
    parser.add_argument("device", type=str, help="c(rown) or m(use)", choices=['c', 'm'])
    parser.add_argument("subjectID", type=int, help="Subject ID")
    parser.add_argument("--sessionName", type=str, help="Session name.  Default 1", default="1")
    parser.add_argument("--ip",
                        default="0.0.0.0", help="The ip to listen on")
    parser.add_argument(
        "--out", type=str, help="Output directory. Default ./output_data/", default="output_data"
    )

    return vars(parser.parse_args())


def main():
    args = parse_args()
    if not os.path.isdir(args["out"]):
        os.makedirs(args["out"])
    if args["sessionName"]:
        filename = f"{args['out']}/{args['subjectID']}_{args['sessionName']}"
    else:
        filename = f"{args['out']}/{args['subjectID']}"
    print(args)
    session = StreamSession()
    device = session.get_device(args, filename)
    device.start_stream()


if __name__ == '__main__':
    main()
