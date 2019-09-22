from __future__ import print_function
import os.path
import warnings
import yaml
from  . import constants as c
from copy import deepcopy


#
# DEFALUTS 
#
_DEFAULTS={
    'is_dev': c.IS_DEV,
    'dry_run': c.DRY_RUN
}


#
# LOAD CONFIG
#
CONFIG=deepcopy(_DEFAULTS)
if os.path.exists(c.TORCH_KIT_CONFIG):
    CONFIG.update(yaml.safe_load(open(c.TORCH_KIT_CONFIG)))


def get(key):
    """ get value
    """
    return CONFIG[key]