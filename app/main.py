import os
import threading
import time

from app.core import middlewares, utils
from app.core.InnerSQLs import InnerSQLite
from app.core.Pipeline import Pipeline
from app.core.pipes.InputPrompt import InputPrompt, WaitPrompt
from app.core.pipes.Mirror import Mirror
from app.core.pipes.SQLRetrieval import SQLRetrieval
from app.core.pipes.VLLMPipe import VLLMPipe
from app.core.utils import list_dict_sum

sql = InnerSQLite('../knowledge.db')


def init_retrieval_pipe():
    # vllm serve meta-llama/Llama-3.2-1B-Instruct --dtype auto --api-key token-abc123
    pipe = SQLRetrieval(
        embedding_path='../post_embeddings.index',
        ids_path='../post_ids.npy',
        sql=sql
    )
    return pipe


def init_llm():
    pipe = VLLMPipe('meta-llama/Llama-3.2-1B-Instruct')
    return pipe


def init():
    pipeline = Pipeline()
    pipeline.queue(InputPrompt("What are searching for:"))
    pipeline.queue(init_retrieval_pipe())
    pipeline.queue(Mirror())
    pipeline.queue(WaitPrompt())
    pipeline.queue(init_llm())
    pipeline.queue(Mirror())
    pipeline.queue(WaitPrompt())
    return pipeline


def init_pipline():
    pipeline = Pipeline()
    pipeline.queue(init_retrieval_pipe())
    pipeline.queue(init_llm())
    pipeline.init()
    return pipeline


# pipeline.loop()
# print(pipeline.execution_times())
# print(list_dict_sum(pipeline.execution_times()))
if __name__ == '__main__':
    pipeline = init_pipline()


    def threaded_process(msg):
        out = pipeline.process(msg, middlewares.contextualize(msg))
        return out


    msgs = [
        'talk about strings',
        'talk about int',
    ]
    threads = []
    for i in range(2):
        process_thread = threading.Thread(target=threaded_process, args=(msgs[i],))
        process_thread.start()
        threads.append(process_thread)

    for thread in threads:
        thread.join()

    print(pipeline.execution_times())
