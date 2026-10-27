"""Current 6DoF predictor. Full and NoRender share identical architecture."""
from .models import Predictor
from .previous_model import pose_loss
__all__ = ['Predictor', 'pose_loss']
