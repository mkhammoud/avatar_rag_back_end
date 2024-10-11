import json
import time
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import List

from app.core.thread_array import ThreadSafeArray


class Pipe(ABC):
    @abstractmethod
    def exec(self, arg) -> any:
        pass

    @abstractmethod
    def end(self):
        pass


class Pipeline:
    def __init__(self, pipes=None):
        self.pipes: List[Pipe] = []
        if pipes is not None:
            [self.queue(p) for p in pipes]
        self.timings = ThreadSafeArray()
        self.round = 0
        self.close = False

    def queue(self, pipe: Pipe):
        self.pipes.append(pipe)

    def process(self, args, middlewares=None):
        if not middlewares:
            middlewares = []
        if not isinstance(middlewares, list):
            middlewares = [middlewares]
        self.timings.append(defaultdict(float))
        result = args
        for pipe in self.pipes:
            if self.close:
                break
            start = time.time()
            for middleware in middlewares:
                result = middleware({'last_result': result, 'current_pipe': pipe})
            result = pipe.exec(result)
            self.timings.get(self.round)[pipe.__class__.__name__] = time.time() - start
        self.round += 1
        return result

    def loop(self):

        def middle_close(args):
            if args == 'stop':
                self.close = True
            return args

        while True:
            if self.close:
                break
            self.process(None, middle_close)

    def execution_times(self):
        return self.timings.array
