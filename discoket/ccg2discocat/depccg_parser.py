from __future__ import annotations

__all__ = ['DepCCGParser', 'DepCCGParseError']

import json
from typing import Any, Iterable, List, Optional, Union

import depccg
import depccg.download
from depccg.parser import EnglishCCGParser
import discopy

from discoket.ccg2discocat.ccg_parser import CCGParser
from discoket.ccg2discocat.ccg_rule import CCGAtomicType
from discoket.ccg2discocat.ccg_tree import CCGTree


class DepCCGParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'depccg failed to parse {repr(self.sentence)}'


class DepCCGParser(CCGParser):
    """CCG parser using depccg as the backend."""

    _unary_rules = [
        ('N', 'NP'),
        ('NP', r'(S[X]/(S[X]\NP))'),
        ('NP', r'((S[X]\NP)\((S[X]\NP)/NP))'),
        ('PP', r'((S[X]\NP)\((S[X]\NP)/PP))'),
        ('NP', r'(((S[X]\NP)/NP)\(((S[X]\NP)/NP)/NP))'),
        ('NP', r'(((S[X]\NP)/PP)\(((S[X]\NP)/PP)/NP))')
    ]

    def __init__(self, *,
                 model: Union[str, EnglishCCGParser] = '',
                 use_model_unary_rules: bool = False,
                 **kwargs: Any) -> None:
        """Instantiate a parser based on `depccg.parser.EnglishCCGParser`.

        Parameters
        ----------
        model : str or depccg.parser.EnglishCCGParser, default: ''
            Can be either:
                - The name of a pre-trained model downloaded by depccg.
                  By default, it uses the "tri_headfirst" model.
                - A pre-instantiated EnglishCCGParser.
        use_model_unary_rules : bool, default: False
            Use the unary rules supplied by the model instead of the
            ones included with `DepCCGParser`.
        kwargs : dict, optional
            Optional arguments passed to `depccg.parser.EnglishCCGParser`.

        Raises
        ------
        TypeError
            If the `model` argument is not of the right type.

        RuntimeError
            If the provided model name is not valid.

        """

        if isinstance(model, EnglishCCGParser):
            self.parser = model
            return
        if not isinstance(model, str):
            raise TypeError('`model` must be an `EnglishCCGParser` or a str.')

        if model in ('basic', 'tri_headfirst'):  # possible aliases
            model = ''
        model_dir, config_file = depccg.download.load_model_directory(
                f'en[{model}]' if model else 'en')

        with open(config_file) as f:
            config = json.load(f)
        if not use_model_unary_rules:
            config['unary_rules'] = self._unary_rules

        self.parser = EnglishCCGParser.from_json(config, model_dir, **kwargs)
        self._last_trees: List[Optional[CCGTree]] = []

    def sentences2trees(
            self,
            sentences: Iterable[str],
            suppress_exceptions: bool = False) -> List[Optional[CCGTree]]:
        sentences = [' '.join(sentence.split()) for sentence in sentences]
        empty_indices = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                if suppress_exceptions:
                    empty_indices.append(i)
                else:
                    raise ValueError('sentence is empty.')

        for i in reversed(empty_indices):
            del sentences[i]

        trees = self._last_trees = []
        if sentences:
            results = self.parser.parse_doc(sentences)
            for result, sentence in zip(results, sentences):
                depccg_tree, score = result[0]
                if score or depccg_tree.word != 'FAILED':
                    trees.append(self._build_ccgtree(depccg_tree))
                elif suppress_exceptions:
                    trees.append(None)
                else:
                    raise DepCCGParseError(sentence)

        for i in empty_indices:
            trees.insert(i, None)

        return trees

    @staticmethod
    def _to_biclosed(cat: depccg.cat.Category) -> discopy.biclosed.Ty:
        """Transform a depccg category into a biclosed type."""

        if not cat.is_functor:
            if cat.is_NorNP:
                return CCGAtomicType.NOUN
            if cat.base == 'S':
                return CCGAtomicType.SENTENCE
            if cat.base == 'PP':
                return CCGAtomicType.PREPOSITION
            if cat.base == 'conj':
                return CCGAtomicType.CONJUNCTION
            if cat.base in ('LRB', 'RRB') or cat.base in ',.:;':
                return CCGAtomicType.PUNCTUATION
        else:
            if cat.slash == '/':
                return (DepCCGParser._to_biclosed(cat.left) <<
                        DepCCGParser._to_biclosed(cat.right))
            if cat.slash == '\\':
                return (DepCCGParser._to_biclosed(cat.right) >>
                        DepCCGParser._to_biclosed(cat.left))
        raise Exception(f'Invalid CCG type: {cat.base}')

    @staticmethod
    def _build_ccgtree(tree: depccg.tree.Tree) -> CCGTree:
        """Transform a depccg derivation tree into a `CCGTree`."""

        if tree.cat.is_type_raised:
            ccg_rule = 'FTR' if tree.cat.is_forward_type_raised else 'BTR'
        elif tree.is_unary:
            ccg_rule = 'U'
        elif tree.is_leaf:
            ccg_rule = 'L'
        else:
            ccg_rule = tree.op_string.upper()
        return CCGTree(
                text=tree.word,
                ccg_rule=ccg_rule,
                biclosed_type=DepCCGParser._to_biclosed(tree.cat),
                children=list(map(DepCCGParser._build_ccgtree, tree.children)))
