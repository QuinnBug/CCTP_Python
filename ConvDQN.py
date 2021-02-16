import numpy as np
import torch as pt
import torch.nn as ptnn
import torch.nn.functional as ptnnf
import torch.optim as pto


class CDQN(ptnn.Module):
