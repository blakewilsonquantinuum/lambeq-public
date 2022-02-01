__all__ = ['__version__', '__version_info__', 'ansatz', 'ccg2discocat',
           'circuit', 'core', 'pregroups', 'reader', 'rewrite', 'tensor',
           'tokeniser']

from lambeq import (ansatz, ccg2discocat, circuit, core, pregroups, reader,
                    rewrite, tensor, tokeniser)
from lambeq.core.utils import is_torch_available

if is_torch_available():
    from lambeq import training
    __all__.append('training')

from lambeq.version import (version as __version__,
                            version_tuple as __version_info__)
