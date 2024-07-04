import abc

class Tracker:

    @abc.abstractmethod
    def tracking(self, _frameData) -> dict:
        '''
        return fullposes        
        '''
        pass
