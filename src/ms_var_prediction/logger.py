import logging

LOG_LEVEL = logging.WARNING

logger = logging.getLogger("markov_switching_model")
logger.setLevel(LOG_LEVEL)

console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)

formatter = logging.Formatter(
    fmt="[%(levelname)s] %(asctime)s %(name)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
