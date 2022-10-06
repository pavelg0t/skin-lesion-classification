from typing import Any
from .factory import TransformBase, TransformFactory

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.transforms import ToTensor


@TransformFactory.register('timm_original')
class TimmTransform(TransformBase):
    """ Transform class from originl timm models """

    def __init__(self, m_name=None, **kwargs):

        self.timm_transform = self._get_timm_transform(m_name)

    def __call__(self, file_path: str, **kwargs: Any) -> Any:
        with open(file_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        return self.timm_transform(img)

    def _get_timm_transform(m_name):

        model = timm.create_model(m_name)
        data_config = resolve_data_config({}, model=model)
        return create_transform(**data_config)


@TransformFactory.register('custom1')
class Custom1Transform(TransformBase):
    """ Custom transform class """

    def __call__(self, file_path: str, **kwargs: Any) -> Any:
        with open(file_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        return ToTensor()(img)
