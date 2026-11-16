'''
icepick: logging setup
'''

# Import external dependencies
import sys
from loguru import logger
from pathlib import Path

# LOG_FORMAT: shared format for the console and file sinks
LOG_FORMAT: str = '<dim>{time:YYYY-MM-DD HH:mm:ss}</> <lvl>[{level}]</> {message}'

# setup: replace all sinks with an INFO console sink and a DEBUG icepick.log in the output directory
def setup(output: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level='INFO')
    logger.add(output / 'icepick.log', format=LOG_FORMAT, level='DEBUG')
