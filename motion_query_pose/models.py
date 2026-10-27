from .v17_models import Predictor as Original
class Predictor(Original):
 def __init__(self,method='Baseline',seed=0):
  # The treatment changes only the loss; preserve MotionQuery inside run().
  if method not in ['Baseline','RenderFull','NoRender','Full']:raise ValueError(method)
  super().__init__('MotionQuery')
  self.loss_variant=method
