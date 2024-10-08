from abc import abstractmethod, ABC
from typing import Callable

from app.core.Pipeline import Pipe
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3


@abstractmethod
class InnerSQL(ABC):
    @abstractmethod
    def init(self) -> any:
        pass

    @abstractmethod
    def query(self, q) -> any:
        pass


class SQLRetrieval(Pipe):
    def __init__(self, embedding_path, ids_path, sql: InnerSQL):
        super().__init__()
        self.embedding_path = embedding_path
        self.ids_path = ids_path
        self.sql = sql
        self.index = None
        self.ids = None
        self.model = None

    def init(self):
        self.sql.init()
        self.index = faiss.read_index(self.embedding_path)
        self.ids = np.load(self.ids_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def exec(self, arg) -> any:
        query_embedding = self.model.encode(arg)
        # D: distances, I: indices of top results
        distance, indices = self.index.search(np.array([query_embedding]), 1)
        result_ids = [self.ids[idx] for idx in indices[0]]
        for post_id in result_ids:
            q = f'select ID, Text from knowledge where ID = {post_id}'
            results = self.sql.query(q)
            if results:
                return results[0][1]
        return None

    def end(self):
        pass
