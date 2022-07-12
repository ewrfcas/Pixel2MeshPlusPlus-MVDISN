from .vgg import VGG16
from .resnet import ResNet18
from .resnet50 import resnet50
from .vgg_tf import VGG16TensorflowAlign, MultiViewVGG16TensorflowAlign


def get_backbone(options):
    if options.backbone == "vgg16":
        nn_encoder = VGG16(pretrained=options.backbone_pretrained)
    elif options.backbone == "vgg16_tf":
        nn_encoder = VGG16TensorflowAlign()
    elif options.backbone == "mv_vgg16_tf":
        nn_encoder = MultiViewVGG16TensorflowAlign()
    elif options.backbone == "res18":
        nn_encoder = ResNet18(pretrained=options.backbone_pretrained)
    elif options.backbone == "res50":
        nn_encoder = resnet50()
    else:
        raise NotImplementedError("No implemented backbone called '%s' found" % options.backbone)
    return nn_encoder
