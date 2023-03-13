import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('example.log', 'w')
formatter = logging.Formatter('%(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)

logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')