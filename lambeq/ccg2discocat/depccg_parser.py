# Copyright 2021 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

__all__ = ['DepCCGParser', 'DepCCGParseError']

import functools
import logging
from typing import Any, List, Optional, TYPE_CHECKING

from discopy.biclosed import Ty

from lambeq.ccg2discocat.ccg_parser import CCGParser
from lambeq.ccg2discocat.ccg_rule import CCGRule
from lambeq.ccg2discocat.ccg_tree import CCGTree
from lambeq.ccg2discocat.ccg_types import CCGAtomicType
from lambeq.core.utils import SentenceBatchType,\
        tokenised_batch_type_check, untokenised_batch_type_check

if TYPE_CHECKING:
    import depccg
    from depccg.annotator import annotate_XX, english_annotator
    from depccg.cat import Category


def _import_depccg() -> None:
    global depccg, Category, annotate_XX, english_annotator
    import depccg
    import depccg.allennlp.utils
    from depccg.annotator import annotate_XX, english_annotator
    from depccg.cat import Category
    import depccg.lang
    import depccg.parsing


# disable irrelevant logging
logging.getLogger('allennlp.common.params').setLevel(logging.ERROR)
logging.getLogger('depccg.chainer.supertagger').setLevel(logging.ERROR)


class DepCCGParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:  # pragma: no cover
        return f'depccg failed to parse: "{self.sentence!r}".'


class DepCCGParser(CCGParser):
    """CCG parser using depccg as the backend."""

    _raw_unary_rules = {'N': ['NP'],
                        'NP': [r'(S[X]/(S[X]\NP))',
                               r'((S[X]\NP)\((S[X]\NP)/NP))',
                               r'(((S[X]\NP)/NP)\(((S[X]\NP)/NP)/NP))',
                               r'(((S[X]\NP)/PP)\(((S[X]\NP)/PP)/NP))'],
                        'PP': [r'((S[X]\NP)\((S[X]\NP)/PP))']}
    _unary_rules = None

    def __init__(self,
                 *,
                 lang: str = 'en',
                 model: Optional[str] = None,
                 use_model_unary_rules: bool = False,
                 annotator: Optional[str] = None,
                 device: int = -1,
                 root_cats: str = 'S[dcl]|S[wq]|S[q]|S[qem]|NP',
                 **kwargs: Any) -> None:
        """Instantiate a parser based on `depccg`.

        Parameters
        ----------
        lang : { 'en', 'ja' }
            The language to use. Use of 'ja' is experimental and has not
            been tested.
        model : str, optional
            The name of the model variant to use, if any.
            (At time of writing) `depccg` supports 'elmo', 'rebank' and
            'elmo_rebank' for English only.
        use_model_unary_rules : bool, default: False
            Use the unary rules supplied by the model instead of the
            ones by `lambeq`.
        annotator : str, optional
            The annotator to use, if any. (At time of writing) `depccg`
            supports 'candc' and 'spacy'.
        device : int, optional
            The ID of the GPU to use. By default, uses the CPU.
        root_cats : str, default: 'S[dcl]|S[wq]|S[q]|S[qem]|NP'
            A bar-separated list of categories allowed at the root of
            the parse.
        **kwargs : dict, optional
            Optional arguments passed to `depccg`.

        """
        _import_depccg()

        depccg.lang.set_global_language_to(lang)
        self.annotator_fun = english_annotator.get(annotator, annotate_XX)
        self.supertagger, config = depccg.instance_models.load_model(model,
                                                                     device)
        (self.apply_binary_rules,
         self.apply_unary_rules,
         self.category_dict,
         _) = depccg.allennlp.utils.read_params(config.config)

        if not use_model_unary_rules:
            if self._unary_rules is None:
                DepCCGParser._unary_rules = {
                        Category.parse(key): [*map(Category.parse, values)]
                        for key, values in self._raw_unary_rules.items()}

            self.apply_unary_rules = functools.partial(
                    depccg.instance_models.GRAMMARS[lang].apply_unary_rules,
                    unary_rules=self._unary_rules
            )

        self.root_categories = [*map(Category.parse, root_cats.split('|'))]
        self.categories: Optional[List[Category]] = None
        self.kwargs = kwargs

        self._last_trees: list[Optional[CCGTree]] = []

    def sentences2trees(
            self,
            sentences: SentenceBatchType,
            suppress_exceptions: bool = False,
            tokenised: bool = False) -> list[Optional[CCGTree]]:
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentences` does not have type '
                                 '`list[list[str]]`.')
            if TYPE_CHECKING:  # temporary fix
                from typing import cast
                sentences = cast(list[list[str]], sentences)
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentences` does not have type '
                                 '`list[str]`.')
            sent_list: list[str] = [str(s) for s in sentences]
            sentences = [sentence.split() for sentence in sent_list]
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
            parses = self._depccg_parse(sentences)
            for (depccg_tree, *_), sentence in zip(parses, sentences):
                if depccg_tree.score > float('-inf'):
                    trees.append(self._build_ccgtree(depccg_tree.tree))
                elif suppress_exceptions:
                    trees.append(None)
                else:
                    raise DepCCGParseError(' '.join(sentence))

        for i in empty_indices:
            trees.insert(i, None)

        return trees

    def _depccg_parse(
            self,
            sentences: list[list[str]]) -> list[list[depccg.tree.ScoredTree]]:
        doc = self.annotator_fun(sentences)
        score_result, categories = self.supertagger.predict_doc(
                [[token.word for token in sentence] for sentence in doc])

        if self.categories is None:
            self.categories = [*map(Category.parse, categories)]

        doc, score_result = depccg.parsing.apply_category_filters(
                doc, score_result, self.categories, self.category_dict)

        return depccg.parsing.run(doc,  # type: ignore[no-any-return]
                                  score_result,
                                  self.categories,
                                  self.root_categories,
                                  self.apply_binary_rules,
                                  self.apply_unary_rules,
                                  **self.kwargs)

    @staticmethod
    def _to_biclosed(cat: Category) -> Ty:
        """Transform a depccg category into a biclosed type."""

        if not cat.is_functor:
            if cat.base in ('N', 'NP'):
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
        biclosed_type = DepCCGParser._to_biclosed(tree.cat)
        if tree.is_leaf:
            children = []
            rule = 'L'
        else:
            children = [*map(DepCCGParser._build_ccgtree, tree.children)]
            if tree.op_string == 'tr':
                rule = ('BTR' if biclosed_type.left.left == biclosed_type.right
                        else 'FTR')
            elif tree.op_symbol == '<un>':
                rule = 'U'
            elif tree.op_string in ('gbx', 'gfc'):
                rule = CCGRule.infer_rule(
                    Ty.tensor(*(child.biclosed_type for child in children)),
                    biclosed_type)
            else:
                rule = tree.op_string.upper()
        return CCGTree(
                text=tree.word,
                rule=rule,
                biclosed_type=biclosed_type,
                children=children)
