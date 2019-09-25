from __future__ import print_function
import os.path
import warnings
import yaml
from copy import deepcopy
#
# CONSTANTS
#
TORCH_KIT_CONFIG='torch_kit.config.yaml'


#
# DEFALUTS 
#
_DEFAULTS={
    'is_dev': False,
    'dry_run': True,
    'noise_reducer': None,
    'poweroff': True,
    'poweroff_wait': 30,
    'print_summary': False,
    'run_key': 'run'
}


#
# LOAD CONFIG
#
CONFIG=deepcopy(_DEFAULTS)
if os.path.exists(TORCH_KIT_CONFIG):
    CONFIG.update(yaml.safe_load(open(TORCH_KIT_CONFIG)))


def get(key):
    """ get value
    """
    return CONFIG[key]