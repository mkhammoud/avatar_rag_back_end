from app.core.Pipeline import Pipe


class Mirror(Pipe):
    def __init__(self, prefix=None):
        self.prefix = prefix if prefix else ''

    def exec(self, arg):
        print(self.prefix, arg)
        return arg

    def end(self):
        pass
