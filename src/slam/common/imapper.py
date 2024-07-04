import abc

class Mapper:

    @abc.abstractmethod
    def update(self, _poses, _frameData):
        pass

    @abc.abstractmethod
    def final(self):
        pass
