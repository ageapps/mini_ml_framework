import numpy
from . import LogManager

class Sequential(object):
    def __init__(self, *args, debug=False):
        self.modules = {}
        for idx, module in enumerate(args):
            self.modules.update({idx: module})
        
        self.logger = LogManager.getLogger(__name__, debug)
        self.logger.debug("Model with:", len(self.modules), "layers")


    def forward (self, x):
        """
        input is x
        output is y
        """ 
        
        xi=x
        for module in self.modules.values():
            xi = module.forward(xi)
        return xi
    
   
    def backward (self, dL_dy) :   
        """
        input is dL/da=dL/dy
        output is dL/dw 
        """ 
        
        yi=dL_dy
        module_list = [self.modules[k] for k in self.modules]
        module_list.reverse()
        for module in module_list:
            yi = module.backward(yi)
        return yi

    
    def update(self, params):
        for i,_ in enumerate(self.modules.values()):
            param = params[i]
            module = self.modules[i]
            if len(param) > 1:
                # only update if module has params
                module.update([param])
    
    def param(self):
        '''
        output is [(Z, dZ/dw)]
        '''
        
        return [module.param()[0] for module in self.modules.values()]
