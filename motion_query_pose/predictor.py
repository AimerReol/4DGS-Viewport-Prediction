from .motion_query_predictor import Predictor as Original
class Predictor(Original):
 def __init__(self,method='Baseline',seed=0):

  if method not in ['Baseline','RenderFull','NoRender','Full']:raise ValueError(method)
  super().__init__('MotionQuery')
  self.loss_variant=method
