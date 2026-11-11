"""Atomic progress snapshots and append-only epoch summaries."""
import json
import math
import os
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta


def _safe(value):
    if isinstance(value, dict):
        return {k:_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class TrainingProgress:
    def __init__(self, directory, epochs, dataset, synthetic=False):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic()
        self.state={'state':'initializing','dataset':dataset,'epochs_total':epochs,
                    'pid':os.getpid(),'synthetic_test':synthetic,'epoch':0,'batch':0,
                    'formal_training_started':not synthetic}
        self.update()

    def update(self, **fields):
        self.state.update(fields)
        self.state['elapsed_seconds']=round(time.monotonic()-self.started,2)
        self.state['updated_at']=datetime.now(timezone(timedelta(hours=8))).isoformat()
        temporary=self.directory/'progress.json.tmp'
        temporary.write_text(json.dumps(_safe(self.state),indent=2,ensure_ascii=False),encoding='utf-8')
        os.replace(temporary,self.directory/'progress.json')

    def epoch_summary(self, record):
        with (self.directory/'history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(_safe(record),ensure_ascii=False)+'\n')
