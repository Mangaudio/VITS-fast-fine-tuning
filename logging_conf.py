import logging
from logging import Filter
import time
import warnings
from logging.handlers import QueueHandler, QueueListener
from torch.multiprocessing import Queue
import tqdm


def suppress_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def init_logging(name: str):
    log_queue = Queue(-1)
    log_filename = f"logs/{name}_{time.strftime('%m-%d_%H:%M:%S')}.log"
    outfile = logging.FileHandler(log_filename, "w")
    outfile.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(levelname)-8s %(asctime)s %(name)-12s]  %(message)s", datefmt="%m-%d %H:%M"
    )
    outfile.setFormatter(formatter)
    logging.getLogger("").addHandler(outfile)

    console = TqdmLoggingHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)-8s %(name)-12s]  %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_filename}")
    listener = QueueListener(log_queue, console, outfile, respect_handler_level=True)
    listener.start()
    return log_queue


class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True


def init_worker_logging(rank: int, log_queue: Queue):
    """
    Method to initialize worker's logging in a distributed setup. The worker processes
    always write their logs to the `log_queue`. Messages in this queue in turn gets picked
    by parent's `QueueListener` and pushes them to respective file/stream log handlers.
    Parameters
    ----------
    rank : ``int``, required
        Rank of the worker
    log_queue: ``Queue``, required
        The common log queue to which the workers
    Returns
    -------
    features : ``np.ndarray``
        The corresponding log power spectrogram.
    """
    queue_handler = QueueHandler(log_queue)

    # Add a filter that modifies the message to put the
    # rank in the log format
    worker_filter = WorkerLogFilter(rank)
    queue_handler.addFilter(worker_filter)

    queue_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)

    # Default logger level is WARNING, hence the change. Otherwise, any worker logs
    # are not going to get bubbled up to the parent's logger handlers from where the
    # actual logs are written to the output
    root_logger.setLevel(logging.INFO)
