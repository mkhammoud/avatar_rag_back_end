import json
import os
import threading
import time

from app.core import middlewares
from app.core.InnerSQLs import InnerSQLite
from app.core.context import AppContext

context = AppContext(InnerSQLite('../flights.db'))
context.init()

pipline = context.get_pipeline()
message = "I want to flight to dubai, I am in doha"
out = pipline.process(message, middlewares.contextualize(message))
print(pipline.execution_times())
