import logging
from logging import Logger

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.Pipeline import Pipeline
from app.core.pipes.AsString import AsString
from app.core.pipes.InputPrompt import WaitPrompt
from app.core.pipes.Mirror import Mirror
from app.core.pipes.SQLRetrieval import SQLRetrieval
from app.core.pipes.VLLMPipe import VLLMPipe


class AppContext:
    def __init__(self, sql):
        self.sql = sql
        self.indexes = None
        self.encoder = None
        self.search_ids = None
        self.logger = logging.getLogger(AppContext.__name__)

    def _init_embeddings(self):
        cursor = self.sql.conn.cursor()
        cursor.execute('SELECT flight_id, description_embedding FROM flight_embeddings')
        rows = cursor.fetchall()
        flight_ids = []
        embeddings = []

        for flight_id, embedding_blob in rows:
            flight_ids.append(flight_id)
            embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)
            embeddings.append(embedding_array)

        return np.array(embeddings).astype('float32'), flight_ids

    def _init_indexes(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])  # Use L2 distance
        index.add(embeddings)
        return index

    def init(self):
        embeddings, self.search_ids = self._init_embeddings()
        self.indexes = self._init_indexes(embeddings)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def get_pipeline(self):
        pipeline = Pipeline()
        pipeline.queue(SQLRetrieval(self, top_k=2))
        pipeline.queue(AsString())
        pipeline.queue(VLLMPipe('meta-llama/Llama-3.2-1B-Instruct'))
        return pipeline
