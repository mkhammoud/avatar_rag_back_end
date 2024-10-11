from app.core.Pipeline import Pipe


class AsString(Pipe):
    def exec(self, arg) -> any:
        return str(arg)

    def end(self):
        pass
