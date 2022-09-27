class Generic():

    def __init__(self, params = {}, exceptions = ['self', '__class__']):
        self._params = []
        for key, value in params.items():
            if key in exceptions: continue
            setattr(self, key, value)
            self._params.append(key)
    
    def __repr__(self):
        object = self.__class__.__name__
        params = [f'{key} = {getattr(self, key)}' for key in self._params]
        return f'{object}({", ".join(params)})'

class AbstractLinearRegression(Generic):

    def __init__(self, params, exceptions = ['self', '__class__']):
        super().__init__(params, exceptions)

    def fit(self):
        raise NotImplementedError()
        
    def predict(self, X):
        return X @ self.w + self.b
