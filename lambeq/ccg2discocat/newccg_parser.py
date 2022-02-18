# Copyright 2022 Cambridge Quantum Computing Ltd.
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

__all__ = ['NewCCGParser', 'NewCCGParseError']

import json
import os
from pathlib import Path
import tarfile
from typing import Any, Optional, Union
from urllib.request import urlretrieve

from discopy.biclosed import Ty

import torch
from transformers import AutoTokenizer

from lambeq.ccg2discocat.ccg_parser import CCGParser
from lambeq.ccg2discocat.ccg_rule import CCGRule
from lambeq.ccg2discocat.ccg_tree import CCGTree
from lambeq.ccg2discocat.ccg_types import CCGAtomicType
from lambeq.ccg2discocat.newccg import (BertForChartClassification, Category,
                                        ChartParser, Grammar, ParseTree,
                                        Sentence, Supertag, Tagger)
from lambeq.core.utils import (SentenceBatchType,
                               tokenised_batch_type_check,
                               untokenised_batch_type_check)

StrPathT = Union[str, 'os.PathLike[str]']

MODELS = {'bert': 'https://qnlp.cambridgequantum.com/models/bert.tar.gz'}


def get_model_dir(model: str, cache_dir: StrPathT = None) -> Path:
    if cache_dir is None:
        try:
            cache_dir = Path(os.getenv('XDG_CACHE_HOME'))
        except TypeError:
            cache_dir = Path.home() / '.cache'
    else:
        cache_dir = Path(cache_dir)
    models_dir = cache_dir / 'lambeq' / 'newccg'
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        raise FileExistsError(f'Cache directory location (`{models_dir}`) '
                              'already exists and is not a directory.')
    return models_dir / model


def download_model(
        model_name: str,
        model_dir: Optional[StrPathT] = None) -> None:  # pragma: no cover
    try:
        url = MODELS[model_name]
    except KeyError:
        raise ValueError(f'Invalid model name : {model_name!r}')

    if model_dir is None:
        model_dir = get_model_dir(model_name)

    def print_progress(chunk: int, chunk_size: int, size: int) -> None:
        percentage = chunk * chunk_size / size
        gb_size = size / 10**9
        print(f'\rDownloading model... {percentage:.1%} of {gb_size:.3} GB',
              end='')

    print('Downloading model...', end='')
    download, headers = urlretrieve(url, reporthook=print_progress)

    print('\nExtracting model...')
    with tarfile.open(download) as tar:
        tar.extractall(model_dir)


class NewCCGParseError(Exception):
    def __init__(self, sentence: str) -> None:
        self.sentence = sentence

    def __str__(self) -> str:
        return f'NewCCG failed to parse {self.sentence!r}.'


class NewCCGParser(CCGParser):
    """CCG parser using NewCCG as the backend."""

    def __init__(self,
                 model_name_or_path: str = 'bert',
                 device: int = -1,
                 cache_dir: Optional[StrPathT] = None,
                 force_download: bool = False,
                 **kwargs: Any) -> None:
        """Instantiate a NewCCGParser.

        Parameters
        ----------
        model_name_or_path : str, default: 'bert'
            Can be either:
                - The path to a directory containing a NewCCG model.
                - The name of a pre-trained model.
                  By default, it uses the "bert" model.
                  See also: `NewCCGParser.available_models()`
        device : int, default: -1
            The GPU device ID on which to run the model, if positive.
            If negative (the default), run on the CPU.
        cache_dir : str or os.PathLike, optional
            The directory to which a downloaded pre-trained model should
            be cached instead of the standard cache
            (`$XDG_CACHE_HOME` or `~/.cache`).
        force_download : bool, default: False
            Force the model to be downloaded, even if it is already
            available locally.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the underlying
            parsers (see Other Parameters). By default, they are set to
            the values in the `pipeline_config.json` file in the model
            directory.

        Other Parameters
        ----------------
        Tagger parameters:
        batch_size : int, optional
            The number of sentences per batch.
        tag_top_k : int, optional
            The maximum number of tags to keep. If 0, keep all tags.
        tag_prob_threshold : float, optional
            The probability multiplier used for the threshold to keep
            tags.
        tag_prob_threshold_strategy : {'relative', 'absolute'}
            If "relative", the probablity threshold is relative to the
            highest scoring tag. Otherwise, the probability is an
            absolute threshold.
        span_top_k : int, optional
            The maximum number of entries to keep per span. If 0, keep
            all entries.
        span_prob_threshold : float, optional
            The probability multiplier used for the threshold to keep
            entries for a span.
        span_prob_threshold_strategy : {'relative', 'absolute'}
            If "relative", the probablity threshold is relative to the
            highest scoring entry. Otherwise, the probability is an
            absolute threshold.

        Chart parser parameters:
        eisner_normal_form : bool, optional
            Whether to use eisner normal form. TODO: explain
        max_parse_trees : int, optional
            A safety limit to the number of parse trees that can be
            generated per parse before automatically failing.
        beam_size : int, optional
            The beam size to use in the chart cells.
        input_tag_score_weight : float, optional
            A scaling multiplier to the log-probabilities of the input
            tags. This means that a weight of 0 causes all of the input
            tags to have the same score.
        missing_cat_score : float, optional
            The default score for a category that is generated but not
            part of the grammar.
        missing_span_score : float, optional
            The default score for a category that is part of the grammar
            but has no score, due to being below the threshold kept by
            the tagger.

        """

        model_dir = Path(model_name_or_path)
        if not model_dir.is_dir():
            model_dir = get_model_dir(model_name_or_path, cache_dir)
            if force_download or not model_dir.is_dir():  # pragma: no cover
                if model_name_or_path not in MODELS:
                    raise ValueError('Invalid model name or path: '
                                     f'{model_name_or_path!r}')
                download_model(model_name_or_path, model_dir)

        with open(model_dir / 'pipeline_config.json') as f:
            config = json.load(f)
        for subconfig in config.values():
            for key in subconfig:
                try:
                    subconfig[key] = kwargs.pop(key)
                except KeyError:
                    pass

        if kwargs:
            raise TypeError('NewCCGParser got unexpected keyword argument(s): '
                            f'{", ".join(map(repr, kwargs))}')

        device_ = torch.device('cpu' if device < 0 else f'cuda:{device}')
        model = (BertForChartClassification.from_pretrained(model_dir)
                                           .eval()
                                           .to(device_))
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tagger = Tagger(model, tokenizer, **config['tagger'])

        grammar = Grammar.load(model_dir / 'grammar.json')
        self.parser = ChartParser(grammar,
                                  self.tagger.model.config.cats,
                                  **config['parser'])

    def sentences2trees(
            self,
            sentences: SentenceBatchType,
            suppress_exceptions: bool = False,
            tokenised: bool = False) -> list[Optional[CCGTree]]:
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentences` does not have type '
                                 '`List[List[str]]`.')
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentences` does not have type '
                                 '`List[str]`.')
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

        trees: list[CCGTree] = []
        if sentences:
            tag_results = self.tagger(sentences)
            tags = tag_results.tags

            for sent in tag_results.sentences:
                words = sent.words
                sent_tags = [[Supertag(tags[id], prob)
                              for id, prob in supertags]
                             for supertags in sent.tags]
                spans = {(start, end): {id: score for id, score in scores}
                         for start, end, scores in sent.spans}

                result = self.parser(Sentence(words, sent_tags, spans))

                try:
                    trees.append(self._build_ccgtree(result[0]))
                except IndexError:
                    if suppress_exceptions:
                        trees.append(None)
                    else:
                        raise NewCCGParseError(' '.join(words))

        for i in empty_indices:
            trees.insert(i, None)

        return trees

    @staticmethod
    def _to_biclosed(cat: Category) -> Ty:
        """Transform a NewCCG category into a biclosed type."""

        if cat.atomic:
            if cat.atom.is_punct:
                return CCGAtomicType.PUNCTUATION
            else:
                atom = str(cat.atom)
                if atom in ('N', 'NP'):
                    return CCGAtomicType.NOUN
                elif atom == 'S':
                    return CCGAtomicType.SENTENCE
                elif atom == 'PP':
                    return CCGAtomicType.PREPOSITION
                elif atom == 'conj':
                    return CCGAtomicType.CONJUNCTION
            raise ValueError(f'Invalid atomic type: {cat.atom!r}')
        else:
            result = NewCCGParser._to_biclosed(cat.result)
            argument = NewCCGParser._to_biclosed(cat.argument)
            return result << argument if cat.fwd else argument >> result

    @staticmethod
    def _build_ccgtree(tree: ParseTree) -> CCGTree:
        """Transform a NewCCG parse tree into a `CCGTree`."""

        children = [NewCCGParser._build_ccgtree(child)
                    for child in filter(None, (tree.left, tree.right))]
        return CCGTree(text=tree.word,
                       rule=CCGRule(tree.rule.name),
                       biclosed_type=NewCCGParser._to_biclosed(tree.cat),
                       children=children)

    @staticmethod
    def available_models() -> list[str]:
        """List the available models."""
        return [*MODELS]
