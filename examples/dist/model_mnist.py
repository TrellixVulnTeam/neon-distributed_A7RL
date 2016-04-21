import logging

from neon.callbacks.callbacks import Callbacks
from neon.data import ArrayIterator, load_mnist
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.models import ModelDist
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, Misclassification
from neon.util.argparser import NeonArgparser

from neon.initializers.initializer import Uniform

class ModelMnist():
  def __init__(self):
    # parse the command line arguments
    parser = NeonArgparser(__doc__)

    self.args = parser.parse_args()

    self.logger = logging.getLogger()
    self.logger.setLevel(self.args.log_thresh)

  def load_data(self):
    # load up the mnist data set
    # split into train and tests sets
    (X_train, y_train), (X_test, y_test), nclass = load_mnist(path=self.args.data_dir)

    # setup a training set iterator
    self.train_set = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(1, 28, 28))
    # setup a validation data set iterator
    self.valid_set = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

  def build_model(self):
    # setup weight initialization function
    init_norm = Gaussian(loc=0.0, scale=0.01)

    # setup model layers
    layers = [Affine(nout=100, init=init_norm, bias=Uniform(), activation=Rectlin()),
              Affine(nout=10, init=init_norm, bias=Uniform(), activation=Logistic(shortcut=True))]

    # setup cost function as CrossEntropy
    self.cost = GeneralizedCost(costfunc=CrossEntropyBinary())
    
    # setup optimizer
    self.optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9, stochastic_round=self.args.rounding)
    
    # initialize model object
    self.model = ModelDist(layers=layers)

  def fit_worker(self):
    # configure callbacks
    #self.callbacks = Callbacks(self.model, eval_set=self.valid_set, **self.args.callback_args)
    # run fit: run inside until finishing
    self.model.fit_worker(self.train_set, optimizer=self.optimizer, 
                          num_epochs=self.args.epochs, cost=self.cost, callbacks=None)

  def fit_ps(self):
    # configure callbacks
    #self.callbacks = Callbacks(self.model, eval_set=self.valid_set, **self.args.callback_args)
    self.model.fit_ps(self.train_set, optimizer=self.optimizer, 
                          num_epochs=self.args.epochs, cost=self.cost, callbacks=None)

  def eval(self):
    print('Misclassification error = %.1f%%' % (self.model.eval(self.valid_set, 
                                                                metric=Misclassification())*100))

    
