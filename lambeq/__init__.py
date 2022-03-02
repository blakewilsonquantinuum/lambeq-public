__all__ = [
        '__version__',
        '__version_info__',

        'ansatz',
        'ccg2discocat',
        'core',
        'pregroups',
        'reader',
        'rewrite',
        'tokeniser',

        'BaseAnsatz',
        'CircuitAnsatz',
        'IQPAnsatz',
        'MPSAnsatz',
        'SpiderAnsatz',
        'Symbol',
        'TensorAnsatz',

        'CCGAtomicType',
        'CCGRule',
        'CCGRuleUseError',
        'CCGTree',

        'CCGParser',
        'CCGBankParseError',
        'CCGBankParser',
        'DepCCGParseError',
        'DepCCGParser',
        'NewCCGParser',
        'NewCCGParseError',
        'WebParseError',
        'WebParser',

        'AtomicType',

        'diagram2str',
        'create_pregroup_diagram',
        'is_pregroup_diagram',

        'Reader',
        'LinearReader',
        'TreeReader',
        'TreeReaderMode',
        'bag_of_words_reader',
        'cups_reader',
        'spiders_reader',
        'stairs_reader',
        'word_sequence_reader',

        'RewriteRule',
        'CoordinationRewriteRule',
        'SimpleRewriteRule',
        'Rewriter',

        'Tokeniser',
        'SpacyTokeniser',
]

from lambeq.core.utils import is_jax_available

from lambeq.version import (version as __version__,
                            version_tuple as __version_info__)

from lambeq import (ansatz, ccg2discocat, core, pregroups, reader, rewrite,
                    tokeniser)

from lambeq.ansatz import (BaseAnsatz, CircuitAnsatz, IQPAnsatz, MPSAnsatz,
                           SpiderAnsatz, Symbol, TensorAnsatz)
from lambeq.ccg2discocat import (
        CCGAtomicType, CCGRule, CCGRuleUseError, CCGTree,
        CCGParser,
        CCGBankParseError, CCGBankParser,
        DepCCGParseError, DepCCGParser,
        NewCCGParseError, NewCCGParser,
        WebParseError, WebParser)
from lambeq.core.types import AtomicType
from lambeq.pregroups import (diagram2str,
                              create_pregroup_diagram, is_pregroup_diagram)
from lambeq.reader import (Reader, LinearReader, TreeReader, TreeReaderMode,
                           bag_of_words_reader, cups_reader, spiders_reader,
                           stairs_reader, word_sequence_reader)
from lambeq.rewrite import (RewriteRule, CoordinationRewriteRule,
                            SimpleRewriteRule, Rewriter)
from lambeq.tokeniser import Tokeniser, SpacyTokeniser

if is_jax_available():
    from lambeq import training
    from lambeq.training import (
            Dataset, Optimiser, SPSAOptimiser,
            Model, NumpyModel, PytorchModel, TketModel,
            Trainer, PytorchTrainer, QuantumTrainer)
    __all__ += [
            'training',

            'Dataset',

            'Optimiser',
            'SPSAOptimiser',

            'Model',
            'NumpyModel',
            'PytorchModel',
            'TketModel',

            'Trainer',
            'PytorchTrainer',
            'QuantumTrainer',
    ]
