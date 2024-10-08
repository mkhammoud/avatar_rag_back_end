from app.core.Pipeline import Pipeline
from app.core.pipes.InputPrompt import InputPrompt
from app.core.pipes.MirrorPrompt import MirrorPrompt


def init():
    pipeline = Pipeline()
    pipeline.queue(InputPrompt("What is your name?:"))
    pipeline.queue(MirrorPrompt())
    return pipeline


def start(pipeline):
    pipeline.start()


if __name__ == '__main__':
    pipeline = init()
    start(pipeline)
