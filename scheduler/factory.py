from .p2m.trainer import P2MTrainer
from .p2m.predictor import P2MPredictor

from .disn.trainer import DISNTrainer
from .disn.predictor_savesdf import DISNPredictor

from .threedgan.trainer import ThreeDGANTrainer
from .threedgan.predictor import ThreeDGANPredictor

from .p2mpp.trainer import P2MPPTrainer
from .p2mpp.predictor import P2MPPPredictor

from .campose.trainer import PoseNetTrainer
from .campose.predictor import PoseNetPredictor

def get_trainer(options, logger, writer):
    if options.model.name == "pixel2mesh":
        trainer = P2MTrainer(options, logger, writer)
    elif options.model.name == "disn":
        trainer = DISNTrainer(options, logger, writer)
    elif options.model.name == "p2mpp":
        trainer = P2MPPTrainer(options, logger, writer)
    elif options.model.name == "campose":
        trainer = PoseNetTrainer(options, logger, writer)
    else:
        raise NotImplementedError("No implemented trainer called '%s' found" % options.model.name)
    return trainer


def get_predictor(options, logger, writer):
    if options.model.name == "pixel2mesh":
        predictor = P2MPredictor(options, logger, writer)
    elif options.model.name == "disn":
        predictor = DISNPredictor(options, logger, writer)
    elif options.model.name == "p2mpp":
        predictor = P2MPPPredictor(options, logger, writer)
    elif options.model.name == "campose":
        predictor = PoseNetPredictor(options, logger, writer)
    else:
        raise NotImplementedError("No implemented trainer called '%s' found" % options.model.name)
    return predictor
