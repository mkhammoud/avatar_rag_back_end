from app.core.Pipeline import Pipe


class Mirror(Pipe):
    def init(self):
        pass

    def exec(self, arg):
        print(arg)
        return arg

    def end(self):
        pass
