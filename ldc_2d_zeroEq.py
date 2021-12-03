from sympy import Symbol, Eq, Abs
import tensorflow as tf

from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain, MonitorDomain, InferenceDomain
from modulus.data import Validation, Monitor, Inference
from modulus.sympy_utils.geometry_2d import Rectangle
from modulus.csv_utils.csv_rw import csv_to_dict
from modulus.PDES import NavierStokes, ZeroEquation
from modulus.node import Node
from modulus.controller import ModulusController

# params for domain
height = 0.1
width = 0.1
vel = 1.5

# define geometry
rec = Rectangle((-width/2, -height/2), (width/2, height/2))
geo = rec

# define sympy varaibles to parametize domain curves
x, y = Symbol('x'), Symbol('y')

# validation data
mapping = {'Points:0': 'x', 'Points:1': 'y', 'U:0': 'u', 'U:1': 'v', 'p': 'p', 'nuT': 'nu'}
openfoam_var = csv_to_dict('openfoam/cavity_uniformVel_zeroEqn_refined.csv', mapping)
openfoam_invar_numpy = {key: value for key, value in openfoam_var.items() if key in ['x', 'y']}
openfoam_outvar_numpy = {key: value for key, value in openfoam_var.items() if key in ['u', 'v', 'nu']}
openfoam_outvar_numpy['nu'] += 1.0e-4

class LDCTrain(TrainDomain):
  def __init__(self, **config):
    super(LDCTrain, self).__init__()
    # top wall
    topWall = geo.boundary_bc(outvar_sympy={'u': vel, 'v': 0},
                              batch_size_per_area=10000,
                              lambda_sympy={'lambda_u': 1.0 - 20*Abs(x), # weight edges to be zero
                                            'lambda_v': 1.0},
                              criteria=Eq(y, height/2))
    self.add(topWall, name="TopWall")

    # no slip
    bottomWall = geo.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                 batch_size_per_area=10000,
                                 criteria=y < height/2)
    self.add(bottomWall, name="NoSlip")

    # interior
    interior = geo.interior_bc(outvar_sympy={'continuity': 0, 'momentum_x': 0, 'momentum_y': 0},
                               bounds={x: (-width/2, width/2), y: (-height/2, height/2)},
                               lambda_sympy={'lambda_continuity': geo.sdf,
                                             'lambda_momentum_x': geo.sdf,
                                             'lambda_momentum_y': geo.sdf},
                               batch_size_per_area=400000)
    self.add(interior, name="Interior")

class LDCVal(ValidationDomain):
  def __init__(self, **config):
    super(LDCVal, self).__init__()
    # valication data from openfoam
    val = Validation.from_numpy(openfoam_invar_numpy, openfoam_outvar_numpy)
    self.add(val, name='Val')

class LDCMonitor(MonitorDomain):
  def __init__(self, **config):
    super(LDCMonitor, self).__init__()
    # metric for mass imbalance, momentum imbalance and peak velocity magnitude
    global_monitor = Monitor(geo.sample_interior(400000, bounds={x: (-width/2, width/2), y: (-height/2, height/2)}),
                             {'mass_imbalance': lambda var: tf.reduce_sum(var['area']*tf.abs(var['continuity'])),
                              'momentum_imbalance': lambda var: tf.reduce_sum(var['area']*tf.abs(var['momentum_x'])+tf.abs(var['momentum_x'])),
                             })
    self.add(global_monitor, 'GlobalMonitor')

class LDCInference(InferenceDomain):
  def __init__(self,**config):
    super(LDCInference,self).__init__()
    #save entire domain
    interior = Inference(geo.sample_interior(1e06, bounds={x: (-width/2, width/2), y: (-height/2, height/2)}), ['u','v','p','nu'])
    self.add(interior, name="Inference")

class LDCSolver(Solver):
  train_domain = LDCTrain
  val_domain = LDCVal
  monitor_domain = LDCMonitor
  inference_domain = LDCInference

  def __init__(self, **config):
    super(LDCSolver, self).__init__(**config)
    self.equations = (NavierStokes(nu='nu', rho=1.0, dim=2, time=False).make_node()
                      + ZeroEquation(nu=1.0e-4, dim=2, time=False, max_distance=0.05).make_node()
                      + [Node.from_sympy(geo.sdf, 'normal_distance')])
    flow_net = self.arch.make_node(name='flow_net',
                                   inputs=['x', 'y'],
                                   outputs=['u', 'v', 'p'])
    self.nets = [flow_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_ldc_2d_zeroEq',
        'start_lr': 3e-4,
        'decay_steps': 20000,
        'max_steps': 1000000
        })

if __name__ == '__main__':
  ctr = ModulusController(LDCSolver)
  ctr.run()
