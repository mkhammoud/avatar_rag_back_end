from app.core.Pipeline import Pipeline
from app.core.pipes.InputPrompt import InputPrompt
from app.core.pipes.Mirror import Mirror


def init():
    pipeline = Pipeline()
    pipeline.queue(InputPrompt("What is your name?:"))
    pipeline.queue(Mirror())
    return pipeline


if __name__ == '__main__':
    pipeline = init()
    pipeline.loop()
