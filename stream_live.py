import argparse
import joblib
import logging
import logging.config
import os
from river import compose
from river import ensemble
from river import preprocessing
from river import tree

from StreamSession import StreamSession

WINDOW_SECONDS = 3

logger = logging.getLogger("file_logger")
subjID = 0

def parse_args():
    parser = argparse.ArgumentParser(
        description="Connects to OSC-streams from portable EEG-devices "
                    "and predicts valence and arousal based on the stream data."
    )
    parser.add_argument("device", type=str, help="c(rown) or m(use)", choices=['c', 'm'])
    parser.add_argument("subjectID", type=int, help="Subject ID")
    parser.add_argument("--modelFileArousal", type=str, help="Path to saved model object if an existing model for "
                                                             "Arousal should be used")
    parser.add_argument("--modelFileValence", type=str, help="Path to saved model object if an existing model for "
                                                             "Valence should be used")
    parser.add_argument("--sessionName", type=str, help="Session name.  Default 1", default="1")
    # IP 0.0.0.0 should get everything that is send to my computer
    parser.add_argument("--ip",
                        default="0.0.0.0", help="The ip to listen on")
    parser.add_argument(
        "--out", type=str, help="Output directory. Default ./output_data/", default="output_data"
    )
    return vars(parser.parse_args())


def load_model(model_file):
    return joblib.load(model_file)


def build_model():
    n_models = 3
    seed = 42
    base_model = tree.HoeffdingAdaptiveTreeClassifier(grace_period=30)  # , drift_window_threshold=50)

    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        # feature_extraction.TargetAgg(by=['Label', 'window'], how=stats.Mean()),
        # tree.HoeffdingAdaptiveTreeClassifier(grace_period=50)
        # tree.HoeffdingAdaptiveTreeClassifier(grace_period=10, seed=seed))
        ensemble.SRPClassifier(model=base_model, n_models=n_models))
    # ensemble.AdaptiveRandomForestClassifier(n_models=3, seed=42, grace_period=30))

    logger.info(model)
    logger.info(f"n_models: {n_models}")

    return model


def main():
    logging.config.fileConfig("./logs.config")
    args = parse_args()
    subjID = args['subjectID']
    if not os.path.isdir(args["out"]):
        os.makedirs(args["out"])
    if args["sessionName"]:
        filename = f"{args['out']}/{args['subjectID']}_{args['sessionName']}"
    else:
        filename = f"{args['out']}/{args['subjectID']}"

    if args["modelFileArousal"]:
        model_a = load_model(args["modelFileArousal"])
    else:
        model_a = build_model()
    if args["modelFileValence"]:
        model_v = load_model(args["modelFileValence"])
    else:
        model_v = build_model()
    logger.info(f"Args: {args}")
    session = StreamSession()
    device = session.get_device(args, filename, model_a=model_a, model_v=model_v, window_sec=WINDOW_SECONDS,
                                logger=logger, live=True)
    device.start_stream()


if __name__ == '__main__':
    main()
