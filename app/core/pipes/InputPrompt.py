from app.core.Pipeline import Pipe


class InputPrompt(Pipe):
    def __init__(self, msg=None):
        super().__init__()
        self.msg = msg

    def init(self):
        pass

    def exec(self, arg):
        return input(self.msg)

    def end(self):
        pass


class WaitPrompt(Pipe):
    def init(self):
        pass

    def exec(self, arg) -> any:
        input('Press Enter continue...')
        return arg

    def end(self):
        pass
