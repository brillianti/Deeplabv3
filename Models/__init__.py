from __future__ import absolute_import


from .Unet import *
from .Deeplabv3 import *

__model_factory = {
    'unet': Unet,
    'Deeplabv3': Deeplabv3
}


def show_avai_models():
    """Displays available models.
    Examples::
        >>> from Models import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(name, num_classes, input_height, input_width):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(
            name, avai_models))
    return __model_factory[name](num_classes,
                                 input_height=input_height,
                                 input_width=input_width)
