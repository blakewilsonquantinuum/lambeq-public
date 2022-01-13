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

__all__ = ['WebParser', 'WebParseError']

import json
from typing import Iterable, List, Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from lambeq.ccg2discocat.ccg_parser import CCGParser
from lambeq.ccg2discocat.ccg_tree import CCGTree

SERVICE_URL = 'https://cqc.pythonanywhere.com/tree/json'


class WebParseError(OSError):
    def __init__(self, sentence: str, error_code: int) -> None:
        self.sentence = sentence
        self.error_code = error_code

    def __str__(self) -> str:
        return (f'Online parsing of sentence {repr(self.sentence)} failed, '
                f'Web status code: {self.error_code}.')


class WebParser(CCGParser):
    """Wrapper that allows passing parser queries to an online interface."""

    def __init__(self, service_url: str = SERVICE_URL) -> None:
        """Initialise a web parser.

        Parameters
        ----------
        service_url : str, default: 'https://cqc.pythonanywhere.com/tree/json'
            The URL to the parser. By default, use CQC's CCG tree
            parser.

        """
        self.service_url = service_url

    def sentences2trees(
            self,
            sentences: Iterable[str],
            suppress_exceptions: bool = False) -> List[Optional[CCGTree]]:
        """Parse multiple sentences into a list of :py:class:`.CCGTree` s.

        Parameters
        ----------
        sentences : iterable of str
            The sentences to be parsed.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.

        Returns
        -------
        list of CCGTree or None
            The parsed trees. (may contain :py:obj:`None` if exceptions
            are suppressed)

        Raises
        ------
        URLError
            If the service URL is not well formed.
        ValueError
            If a sentence is blank.
        WebParseError
            If the parser fails to obtain a parse tree from the server.

        """

        sentences = [' '.join(sentence.split()) for sentence in sentences]
        empty_indices = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                if suppress_exceptions:
                    empty_indices.append(i)
                else:
                    raise ValueError(f'Sentence at index {i} is blank.')

        for i in reversed(empty_indices):
            del sentences[i]

        trees: List[Optional[CCGTree]] = []
        for sent in sentences:
            params = urlencode({'sentence': sent})
            url = f'{self.service_url}?{params}'

            try:
                with urlopen(url) as f:
                    data = json.load(f)
            except HTTPError as e:
                if suppress_exceptions:
                    tree = None
                else:
                    raise WebParseError(sentence, e.code)
            else:
                tree = CCGTree.from_json(data)
            trees.append(tree)

        for i in empty_indices:
            trees.insert(i, None)

        return trees
