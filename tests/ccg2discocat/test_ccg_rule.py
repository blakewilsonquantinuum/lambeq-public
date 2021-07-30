import pytest

from discopy import biclosed, Word
from discopy.biclosed import Box
from discopy.rigid import Cap, Cup, Id, Swap, caps

from discoket.ccg2discocat.ccg_rule import CCGRuleUseError, CCGRule, RPL, RPR
from discoket.ccg2discocat.ccg_types import CCGAtomicType
from discoket.core.types import AtomicType

from discoket.ccg2discocat.ccg_tree import CCGTree, PlanarBX, PlanarFX


N = AtomicType.NOUN
P = AtomicType.PREPOSITION
S = AtomicType.SENTENCE

i = biclosed.Ty()
n = CCGAtomicType.NOUN
p = CCGAtomicType.PREPOSITION
punc = CCGAtomicType.PUNCTUATION
s = CCGAtomicType.SENTENCE

comma = CCGTree(',', biclosed_type=punc)
_and = CCGTree(text='and', biclosed_type=CCGAtomicType.CONJUNCTION)
be = CCGTree('be', biclosed_type=s << n)
do = CCGTree('do', biclosed_type=s << s)
_is = CCGTree('is', biclosed_type=n >> s)
it = CCGTree('it', biclosed_type=n)
_not = CCGTree(text='not', biclosed_type=s >> s)
the = CCGTree('the', biclosed_type=n << n)


class CCGRuleTester:
    tree = None
    biclosed_diagram = None
    diagram = None
    planar_biclosed_diagram = None
    planar_diagram = None

    def test_biclosed_diagram(self):
        assert self.tree.to_biclosed_diagram() == self.biclosed_diagram

    def test_diagram(self):
        assert self.tree.to_diagram() == self.diagram

    def test_planar_biclosed_diagram(self):
        diagram = self.planar_biclosed_diagram or self.biclosed_diagram
        assert self.tree.to_biclosed_diagram(planar=True) == diagram

    def test_planar_diagram(self):
        diagram = self.planar_diagram or self.diagram
        assert self.tree.to_diagram(planar=True) == diagram


class TestBackwardApplication(CCGRuleTester):
    tree = CCGTree(rule='BA', biclosed_type=s, children=(it, _is))

    biclosed_words = Box('it', i, n) @ Box('is', i, n >> s)
    biclosed_diagram = biclosed_words >> biclosed.BA(n >> s)

    words = Word('it', N) @ Word('is', N >> S)
    diagram = words >> (Cup(N, N.r) @ Id(S))


class TestBackwardComposition(CCGRuleTester):
    tree = CCGTree(rule='BC', biclosed_type=n >> s, children=(_is, _not))

    # biclosed diagram
    biclosed_words = Box('is', i, n >> s) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> biclosed.BC(n >> s, s >> s)

    # rigid diagram
    words = Word('is', N >> S) @ Word('not', S >> S)
    diagram = words >> (Id(N.r) @ Cup(S, S.r) @ Id(S))


class TestBackwardCrossedComposition(CCGRuleTester):
    tree = CCGTree(rule='BX', biclosed_type=s << n, children=(be, _not))

    biclosed_words = Box('be', i, s << n) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> biclosed.BX(s << n, s >> s)

    words = Word('be', S << N) @ Word('not', S >> S)
    diagram = (words >>
               Id(S) @ Swap(N.l, S.r) @ Id(S) >>
               Cup(S, S.r) @ Swap(N.l, S))

    be_box, not_box = biclosed_words.boxes
    planar_biclosed_diagram = be_box >> PlanarBX(s << n, not_box)

    be_word, not_word = words.boxes
    planar_diagram = (be_word >>
                      Id(S) @ not_word @ Id(N.l) >>
                      Cup(S, S.r) @ Id(S << N))


class TestBackwardTypeRaising(CCGRuleTester):
    tree = CCGTree(rule='BTR', biclosed_type=(s << n) >> s, children=(it,))

    biclosed_diagram = (Box('it', i, n) >>
                        biclosed.Curry(biclosed.FA(s << n), left=True))

    diagram = (Word('it', N) >>
               Cap(N, N.l) @ Id(N) >>
               Id(N) @ Cap(S.r, S) @ Cup(N.l, N))


class TestConjunctionLeft(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n >> n, children=(_and, it))

    biclosed_words = Box('and', i, (n >> n) << n) @ Box('it', i, n)
    biclosed_diagram = biclosed_words >> biclosed.FA((n >> n) << n)

    words = Word('and', N >> N << N) @ Word('it', N)
    diagram = words >> (Id(N >> N) @ Cup(N.l, N))


class TestConjunctionRight(CCGRuleTester):
    tree = CCGTree(rule='CONJ', biclosed_type=n << n, children=(it, _and))

    biclosed_words = Box('it', i, n) @ Box('and', i, n >> (n << n))
    biclosed_diagram = biclosed_words >> biclosed.BA(n >> (n << n))

    words = Word('it', N) @ Word('and', N >> N << N)
    diagram = words >> (Cup(N, N.r) @ Id(N << N))


def test_conjunction_error():
    tree = CCGTree(rule='CONJ', biclosed_type=n, children=(it, it))
    with pytest.raises(CCGRuleUseError):
        tree.to_biclosed_diagram()


class TestForwardApplication(CCGRuleTester):
    tree = CCGTree(rule='FA', biclosed_type=s, children=(be, it))

    biclosed_words = Box('be', i, s << n) @ Box('it', i, n)
    biclosed_diagram = biclosed_words >> biclosed.FA(s << n)

    words = Word('be', S << N) @ Word('it', N)
    diagram = words >> (Id(S) @ Cup(N.l, N))


class TestForwardComposition(CCGRuleTester):
    tree = CCGTree(rule='FC', biclosed_type=s << n, children=(be, the))

    biclosed_words = Box('be', i, s << n) @ Box('the', i, n << n)
    biclosed_diagram = biclosed_words >> biclosed.FC(s << n, n << n)

    words = Word('be', S << N) @ Word('the', N << N)
    diagram = words >> (Id(S) @ Cup(N.l, N) @ Id(N.l))


class TestForwardCrossedComposition(CCGRuleTester):
    tree = CCGTree(rule='FX', biclosed_type=s >> s, children=(do, _not))

    biclosed_words = Box('do', i, s << s) @ Box('not', i, s >> s)
    biclosed_diagram = biclosed_words >> biclosed.FX(s << s, s >> s)

    words = Word('do', S << S) @ Word('not', S >> S)
    diagram = (words >>
               Id(S) @ Swap(S.l, S.r) @ Id(S) >>
               Swap(S, S.r) @ Cup(S.l, S))

    do_box, not_box = biclosed_words.boxes
    planar_biclosed_diagram = not_box >> PlanarFX(s >> s, do_box)

    do_word, not_word = words.boxes
    planar_diagram = (not_word >>
                      Id(S.r) @ do_word @ Id(S) >>
                      Id(S >> S) @ Cup(S.l, S))


class TestForwardTypeRaising(CCGRuleTester):
    tree = CCGTree(rule='FTR', biclosed_type=s << (n >> s), children=(it,))

    biclosed_diagram = Box('it', i, n) >> biclosed.Curry(biclosed.BA(n >> s))

    diagram = (Word('it', N) >>
               Id(N) @ caps(N >> S, (N >> S).l) >>
               Cup(N, N.r) @ Id((S << S) @ N))


class TestLexical(CCGRuleTester):
    tree = it

    biclosed_diagram = Box('it', i, n)

    diagram = Word('it', N)

    def test_rule_use_error(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.rule(i, i)


class TestRemovePunctuationLeft(CCGRuleTester):
    tree = CCGTree(rule='LP', biclosed_type=n, children=(comma, it))

    biclosed_words = Box(',', i, punc) @ Box('it', i, n)
    biclosed_diagram = biclosed_words >> RPL(punc, n)

    diagram = Word('it', N)


class TestRemovePunctuationRight(CCGRuleTester):
    tree = CCGTree(rule='RP', biclosed_type=n, children=(it, comma))

    biclosed_words = Box('it', i, n) @ Box(',', i, punc)
    biclosed_diagram = biclosed_words >> RPR(n, punc)

    diagram = Word('it', N)


class TestUnary(CCGRuleTester):
    tree = CCGTree(rule='U', biclosed_type=s, children=(be,))

    biclosed_diagram = Box('be', i, s)

    diagram = Word('be', S)


class TestUnknown(CCGRuleTester):
    tree = CCGTree(rule='UNK', biclosed_type=i, children=[the])

    def test_biclosed_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_biclosed_diagram()

    def test_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_diagram()

    def test_planar_biclosed_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_biclosed_diagram(planar=True)

    def test_planar_diagram(self):
        with pytest.raises(CCGRuleUseError):
            self.tree.to_diagram(planar=True)

    def test_initialisation(self):
        assert CCGRule('missing') == CCGRule.UNKNOWN
