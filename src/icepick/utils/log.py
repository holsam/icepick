'''
icepick: logging setup
'''

# Import external dependencies
import sys
from loguru import logger
from pathlib import Path

# LOG_FORMAT: shared format for the console and file sinks
LOG_FORMAT: str = '<dim>{time:YYYY-MM-DD HH:mm:ss}</> <lvl>[{level}]</>{extra[pad]} [{extra[tomogram]}] {message}'

# PROGRESS: custom level between DEBUG (10) and INFO (20), registered on import
logger.level('PROGRESS', no=15, color='<cyan><bold>')

# CONSOLE_LEVELS: console level by number of -v flags
CONSOLE_LEVELS: tuple[str, ...] = ('INFO', 'PROGRESS', 'DEBUG')

# progress: log a message at the PROGRESS level
def progress(message: str) -> None:
    logger.opt(depth=1).log('PROGRESS', message)

# _pad: patcher adding spaces after level so messages align
def _pad(record) -> None:
    record['extra']['pad'] = ' ' * (len('PROGRESS') - len(record['level'].name))

# setup: replace all sinks with a console sink (INFO, PROGRESS at -v, DEBUG at -vv) and a DEBUG icepick.log in the output directory
def setup(output: Path, tomogram: str, *, verbosity: int = 0) -> None:
    logger.configure(extra={'tomogram': tomogram}, patcher=_pad)
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level=CONSOLE_LEVELS[min(verbosity, 2)])
    logger.add(output / 'icepick.log', format=LOG_FORMAT, level='DEBUG')

# error_exit: log an error message then exit with error code 1
def error_exit(msg: str, exc) -> None:
    logger.error(msg)
    sys.stderr.write('\n')
    raise SystemExit(1) from exc
