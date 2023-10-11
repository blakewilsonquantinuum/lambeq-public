import pytest

from discopy.grammar.pregroup import (Box, Cap, Cup, Diagram, Id, Ob, Spider,
                                      Swap, Ty, Word)

from lambeq import (AtomicType, Rewriter, CoordinationRewriteRule,
                    CurryRewriteRule, RemoveCupsRewriter,
                    RemoveSwapsRewriter, SimpleRewriteRule, stairs_reader,
                    UnifyCodomainRewriter, UnknownWordsRewriteRule)

N = AtomicType.NOUN
S = AtomicType.SENTENCE


def test_initialisation():
    assert (Rewriter().rules == Rewriter([Rewriter._available_rules[rule]
            for rule in Rewriter._default_rules]).rules)

    assert all([rule in Rewriter.available_rules() for rule in Rewriter._default_rules])
    with pytest.raises(ValueError):
        Rewriter(['nonexistent rule'])


def test_custom_rewriter():
    placeholder = SimpleRewriteRule.placeholder(S)
    not_box = Box('NOT', S, S)
    not_rewriter = SimpleRewriteRule(
        cod=S, template=placeholder >> not_box)

    diagram = Word('I think', S)
    assert not_rewriter(diagram) == diagram >> not_box


def test_auxiliary():
    cod = (N >> S) << (N >> S)
    we = Word('we', N)
    go = Word('go', N >> S)
    cups = Cup(N, N.r) @ Id(S) @ Diagram.cups((N >> S).l, N >> S)

    diagram = (we @ Word('will', cod) @ go) >> cups
    expected_diagram = (we @ Diagram.caps(cod[:2], cod[2:]) @ go) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['auxiliary'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_connector():
    left_words = Word('I', N) @ Word('hope', N >> S << S)
    right_words = Word('this', N) @ Word('succeeds', N >> S)
    cups = (Cup(N, N.r) @ Id(S) @ Cup(S.l, S) @
            Diagram.cups((N >> S).l, N >> S))

    diagram = (left_words @ Word('that', S << S) @ right_words) >> cups
    expected_diagram = (left_words @ Cap(S, S.l) @ right_words) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['connector'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_determiner():
    book = Word('book', N)
    cups = Id(N) @ Cup(N.l, N)

    diagram = (Word('the', N << N) @ book) >> cups
    expected_diagram = (Cap(N, N.l) @ book) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['determiner'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_postadverb():
    cod = (N >> S) >> (N >> S)
    vp = Word('we', N) @ Word('go', N >> S)
    cups = Diagram.cups(cod[:3].l, cod[:3]) @ Id(S)

    diagram = (vp @ Word('quickly', cod)) >> cups
    expected_diagram = (vp @ (Word('quickly', S >> S) >>
                              Id(S.r) @ Cap(N.r.r, N.r) @ Id(S))) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['postadverb'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_preadverb():
    we = Word('we', N)
    go = Word('go', N >> S)
    cups = Cup(N, N.r) @ Id(S) @ Diagram.cups((N >> S).l, N >> S)

    diagram = (we @ Word('quickly', (N >> S) << (N >> S)) @ go) >> cups
    expected_diagram = (we @ (Cap(N.r, N) >>
                              Id(N.r) @ Word('quickly', S << S) @ Id(N)) @
                        go) >> cups

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['preadverb'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_prepositional_phrase():
    cod = (N >> S) >> (N >> S << N)
    vp = Word('I', N) @ Word('go', N >> S)
    bed = Word('bed', N)
    cups = Diagram.cups(cod[:3].l, cod[:3]) @ Id(S) @ Cup(N.l, N)

    diagram = (vp @ Word('to', cod) @ bed) >> cups
    expected_diagram = (vp @ (Word('to', S >> S << N) >>
                              Id(S.r) @ Cap(N.r.r, N.r) @ Id(S << N)) @ bed >>
                        cups)

    assert Rewriter([])(diagram) == diagram
    assert Rewriter(['prepositional_phrase'])(diagram) == expected_diagram
    assert Rewriter()(diagram) == expected_diagram


def test_rel_pronoun():
    cows = Word('cows', N)
    that_subj = Word('that', N.r @ N @ S.l @ N)
    that_obj = Word('that', N.r @ N @ N.l.l @ S.l)
    eat = Word('eat', N >> S << N)
    grass = Word('grass', N)

    rewriter = Rewriter(['subject_rel_pronoun', 'object_rel_pronoun'])

    diagram_subj = Id().tensor(cows, that_subj, eat, grass)
    diagram_subj >>= Cup(N, N.r) @ N @ Diagram.cups(S.l @ N, N.r @ S) @ Cup(N.l, N)

    expected_diagram_subj = Diagram.decode(
            dom=Ty(), cod=N,
            boxes=[cows, Spider(1, 2, N), Spider(0, 1, S.l), eat, Cup(N, N.r),
                   Cup(S.l, S), grass, Cup(N.l, N)],
            offsets=[0, 0, 1, 3, 2, 1, 2, 1])

    assert rewriter(diagram_subj).normal_form() == expected_diagram_subj

    diagram_obj = Id().tensor(grass, that_obj, cows, eat)
    diagram_obj >>= Cup(N, N.r) @ Id(N) @ Id(N.l.l @ S.l) @ Cup(N, N.r) @ Id(S @ N.l)
    diagram_obj >>= Id(N) @ Diagram.cups(N.l.l @ S.l, S @ N.l)

    expected_diagram_obj = Diagram.decode(
            dom=Ty(), cod=N,
            boxes=[grass, Spider(1, 2, N), Cap(N.l, N.l.l), Swap(N.l, N.l.l),
                   Spider(0, 1, S.l), cows, eat, Cup(N, N.r), Cup(S.l, S),
                   Cup(N.l.l, N.l), Cup(N.l, N)],
            offsets=[0, 0, 1, 1, 2, 3, 4, 3, 2, 1, 1])

    assert rewriter(diagram_obj).normal_form() == expected_diagram_obj


def test_coordination():
    eggs = Word('eggs', N)
    ham = Word('ham', N)

    words = eggs @ Word('and', N >> N << N) @ ham
    cups = Cup(N, N.r) @ Id(N) @ Cup(N.l, N)

    rewriter = Rewriter([CoordinationRewriteRule()])
    diagram = words >> cups
    expected_diagram = eggs @ ham >> Spider(2, 1, N)

    assert rewriter(diagram).normal_form() == expected_diagram


def test_curry_functor():
    n, s = map(Ty, 'ns')
    diagram = (
        Word('I', n) @ Word('see', n.r @ s @ n.l) @
        Word('the', n @ n.l) @ Word('truth', n)).cup(5, 6).cup(3, 4).cup(0, 1)
    expected = (Word('I', n) @ Word('truth', n) >> Id(n) @ Box('the', n, n)
                >> Box('see', n @ n, s))

    rewriter = Rewriter([CurryRewriteRule()])
    assert rewriter(diagram).normal_form() == expected


def test_merge_wires_rewriter():
    n, s = map(Ty, 'ns')
    take = Word('take', n.r @ s @ n.l)
    the = Word('the', n @ n.l)
    bus = Word('bus', n)

    diagram = (take @ the @ bus).cup(4, 5).cup(2, 3)
    expected_diagram = diagram >> Box('MERGE_n.r @ s', n.r @ s, s)

    rewriter = UnifyCodomainRewriter()

    assert rewriter(diagram) == expected_diagram


def test_remove_cups_rewriter():
    n, s = map(Ty, 'ns')

    remove_cups = RemoveCupsRewriter()

    d1 = (
        Word("box1", n @ n @ s) @ Word("box2", (n @ s).r)
        >> Id(n) @ Diagram.cups(n @ s, (n @ s).r))
    expect_d1 = (
        Word("box1", n @ n @ s) >> Id(n) @ Word("box2", (n @ s).r).l)
    assert remove_cups(d1) == expect_d1

    d2 = (
        Word("box1", n @ s) @ Word("box2", s.r @ n) @ Word("box3", n.r @ n.r)
        >> Id(n) @ Cup(s, s.r) @ Cup(n, n.r) @ Id(n.r) >> Cup(n, n.r))
    expect_d2 = (
        Word("box3", n.r @ n.r) >> Id(n.r) @ Cap(s.r.r, s.r) @ Id(n.r)
        >> Word("box2", s.r @ n).r @ Word("box1", n @ s).r)
    assert remove_cups(d2) == expect_d2

    d3 = (
        (Word("box1", n) >> Spider(1, 2, n)) @ Word("box2", n.r @ n.r @ n)
        >> Diagram.cups(n @ n, n.r @ n.r) @ Id(n))
    expect_d3 = (
        Word("box2", n.r @ n.r @ n)
        >> (Word("box1", n) >> Spider(1, 2, n)).r @ Id(n)
    )
    assert remove_cups(d3) == expect_d3

    # test disconnected
    assert (remove_cups(Id().tensor(d1, d2, d3))
            == Id().tensor(*map(remove_cups, (d1, d2, d3))))

    # test illegal cups
    assert remove_cups(d1.r.dagger()) == remove_cups(d1).r.dagger()
    assert remove_cups(d3.l.dagger()) == remove_cups(d3).l.dagger()

    # scalars can be bent both ways
    assert remove_cups(d2.r.dagger()) == remove_cups(d2).dagger().normal_form()

    d4 = (
        Word('box1', n) @ Word('box2', n) @ Word('box3', n.r @ n.r @ n)
        >> Diagram.cups(n @ n, (n @ n).r) @ Id(n))
    expect_d4 = (
        Word('box3', n.r @ n.r @ n) >>
        (Word('box1', n) @ Word('box2', n)).r @ Id(n))
    assert remove_cups(d4) == expect_d4

    assert remove_cups(d4 @ Id(n @ s) @ d4) == expect_d4 @ Id(n @ s) @ expect_d4

    type_raised = (
        Diagram.caps(s @ n @ s, (s @ n @ s).l) @ Word('w1', s) @ Word('w2', n @ s)
        >> Id(s @ n @ s) @ Diagram.cups((s @ n @ s).l, s @ n @ s))

    def remove_caps(diagram):
        return remove_cups(diagram.dagger()).dagger()
    assert remove_caps(remove_cups(type_raised)) == Word('w1', s) @ Word('w2', n @ s)


def test_remove_swaps_rewriter():
    n, s = map(Ty, 'ns')
    remove_swaps = RemoveSwapsRewriter()

    inp_diagr = Diagram.decode(
        dom=Ty(),
        cod=s,
        boxes=[Word('I', Ty('n')), Word('do', Ty(Ob('n', z=1), 's', Ob('s', z=-1), 'n')),
                Word('not', Ty(Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 's')),
                Word('run', Ty(Ob('n', z=1), 's')), Swap(Ty('n'), Ty(Ob('s', z=1))),
                Swap(Ty(Ob('s', z=-1)), Ty(Ob('s', z=1))), Cup(Ty('s'), Ty(Ob('s', z=1))),
                Swap(Ty('n'), Ty(Ob('n', z=2))), Swap(Ty(Ob('s', z=-1)), Ty(Ob('n', z=2))),
                Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))), Swap(Ty('n'), Ty(Ob('n', z=1))),
                Swap(Ty(Ob('s', z=-1)), Ty(Ob('n', z=1))), Cup(Ty('n'), Ty(Ob('n', z=1))),
                Swap(Ty('n'), Ty('s')), Swap(Ty(Ob('s', z=-1)), Ty('s')),
                Cup(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty(Ob('s', z=-1)), Ty('s'))],
            offsets= [0, 1, 5, 9, 4, 3, 2, 3, 2, 1, 2, 1, 0, 1, 0, 2, 1]
    )

    out_diagr = Diagram.decode(
        dom=Ty(),
        cod=s,
        boxes=[Word('I', Ty('n')),
                Word('do not', Ty(Ob('n', z=1), 's', Ob('s', z=-1), 'n')),
                Word('run', Ty(Ob('n', z=1), 's')),
                Cup(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty('n'),
                Ty(Ob('n', z=1))),
                Cup(Ty(Ob('s', z=-1)), Ty('s'))],
        offsets=[0, 1, 5, 0, 2, 1]
    )

    assert remove_swaps(inp_diagr) == out_diagr


def test_remove_swaps_rewriter_cross_composition():
    n, s = map(Ty, 'ns')
    remove_swaps = RemoveSwapsRewriter()

    inp_diagr = Diagram.decode(
        dom=Ty(),
        cod=s,
        boxes=[Word('I', Ty('n')), Word('do', Ty(Ob('n', z=1), 's', Ob('s', z=-1), 'n')),
                Word('not', Ty(Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 's')),
                Word('run', Ty(Ob('n', z=1), 's')), Swap(Ty('n'), Ty(Ob('s', z=1))),
                Swap(Ty(Ob('s', z=-1)), Ty(Ob('s', z=1))), Cup(Ty('s'), Ty(Ob('s', z=1))),
                Swap(Ty('n'), Ty(Ob('n', z=2))), Swap(Ty(Ob('s', z=-1)), Ty(Ob('n', z=2))),
                Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))), Swap(Ty('n'), Ty(Ob('n', z=1))),
                Swap(Ty(Ob('s', z=-1)), Ty(Ob('n', z=1))), Cup(Ty('n'), Ty(Ob('n', z=1))),
                Swap(Ty('n'), Ty('s')), Swap(Ty(Ob('s', z=-1)), Ty('s')),
                Cup(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty(Ob('s', z=-1)), Ty('s'))],
            offsets= [0, 1, 5, 9, 4, 3, 2, 3, 2, 1, 2, 1, 0, 1, 0, 2, 1]
    )

    out_diagr = Diagram.decode(
        dom=Ty(),
        cod=s,
        boxes=[Word('I', Ty('n')),
                Word('do not', Ty(Ob('n', z=1), 's', Ob('s', z=-1), 'n')),
                Word('run', Ty(Ob('n', z=1), 's')),
                Cup(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty('n'),
                Ty(Ob('n', z=1))),
                Cup(Ty(Ob('s', z=-1)), Ty('s'))],
        offsets=[0, 1, 5, 0, 2, 1]
    )

    assert remove_swaps(inp_diagr) == out_diagr


def test_remove_swaps_rewriter_cross_comp_and_unary_rule():
    n, s = map(Ty, 'ns')
    remove_swaps = RemoveSwapsRewriter()

    inp_diagr = Diagram.decode(
        dom=Ty(),
        cod=n,
        boxes=[Word('The', Ty('n', Ob('n', z=-1))),
               Word('best', Ty('n', Ob('n', z=-1))), Word('film', Ty('n')),
               Word('I', Ty('n')), Word("'ve", Ty(Ob('n', z=1), 's', Ob('s', z=-1), 'n')),
               Word('ever', Ty(Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 'n')),
               Word('seen', Ty(Ob('n', z=1), 's', Ob('n', z=1))),
               Cup(Ty(Ob('n', z=-1)), Ty('n')), Cup(Ty(Ob('n', z=-1)), Ty('n')),
               Swap(Ty('n'), Ty(Ob('s', z=1))), Swap(Ty(Ob('s', z=-1)), Ty(Ob('s', z=1))),
               Cup(Ty('s'), Ty(Ob('s', z=1))), Swap(Ty('n'), Ty(Ob('n', z=2))),
               Swap(Ty(Ob('s', z=-1)), Ty(Ob('n', z=2))), Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))),
               Swap(Ty('n'), Ty(Ob('n', z=1))), Swap(Ty(Ob('s', z=-1)), Ty(Ob('n', z=1))),
               Cup(Ty('n'), Ty(Ob('n', z=1))), Swap(Ty('n'), Ty('n')), Swap(Ty(Ob('s', z=-1)), Ty('n')),
               Cup(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty(Ob('s', z=-1)), Ty('s')),
               Swap(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty('n'), Ty(Ob('n', z=1)))],
        offsets=[0, 2, 4, 5, 6, 10, 14, 1, 1, 5, 4, 3, 4, 3, 2,
                 3, 2, 1, 2, 1, 3, 2, 1, 0]
    )

    out_diagr = Diagram.decode(
        dom=Ty(),
        cod=n,
        boxes=[Word('The', Ty('n', Ob('n', z=-1))), Word('best', Ty('n', Ob('n', z=-1))),
                Word('film', Ty('n')), Word('I', Ty('n')),
                Word("'ve ever", Ty(Ob('n', z=1), Ob('n', z=1), Ob('s', z=-1), 'n')),
                Word('seen', Ty(Ob('n', z=1), 's', 'n')), Cup(Ty(Ob('n', z=-1)),
                Ty('n')), Cup(Ty(Ob('n', z=-1)), Ty('n')), Cup(Ty('n'), Ty(Ob('n', z=1))),
                Cup(Ty('n'), Ty(Ob('n', z=1))), Cup(Ty(Ob('s', z=-1)), Ty('s')),
                Cup(Ty('n'), Ty(Ob('n', z=1)))],
        offsets=[0, 2, 4, 5, 6, 10, 1, 1, 1, 3, 2, 0]
    )

    assert remove_swaps(inp_diagr) == out_diagr


def test_remove_swaps_rewriter_shorten_type():
    n, s = map(Ty, 'ns')
    remove_swaps = RemoveSwapsRewriter()

    inp_diagr = Diagram.decode(
        dom=Ty(),
        cod=n,
        boxes=[Word('What', Ty('n', Ob('n', z=-2), Ob('s', z=-1))), Word('Alice', Ty('n')),
               Word('is', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
               Word('and', Ty('n', Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 's',
                               Ob('n', z=-1), Ob('n', z=-2), Ob('s', z=-1), 'n')),
               Word('is', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
               Word('not', Ty(Ob('s', z=1), Ob('n', z=2), Ob('n', z=1), 's')),
               Cup(Ty(Ob('n', z=-1)), Ty('n')), Cup(Ty('s'), Ty(Ob('s', z=1))),
               Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))), Cup(Ty('n'), Ty(Ob('n', z=1))),
               Cup(Ty(Ob('s', z=-1)), Ty('s')), Cup(Ty(Ob('n', z=-2)), Ty(Ob('n', z=-1))),
               Swap(Ty(Ob('n', z=-1)), Ty(Ob('s', z=1))), Cup(Ty('s'), Ty(Ob('s', z=1))),
               Swap(Ty(Ob('n', z=-1)), Ty(Ob('n', z=2))), Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))),
               Swap(Ty(Ob('n', z=-1)), Ty(Ob('n', z=1))), Cup(Ty('n'), Ty(Ob('n', z=1))),
               Swap(Ty(Ob('n', z=-1)), Ty('s')), Cup(Ty(Ob('s', z=-1)), Ty('s')),
               Cup(Ty(Ob('n', z=-2)), Ty(Ob('n', z=-1)))],
        offsets=[0, 3, 4, 7, 16, 19, 6, 5, 4, 9, 8, 7, 6, 5, 5, 4, 4,
                 3, 3, 2, 1]
    )

    out_diagr = Diagram.decode(
        dom=Ty(),
        cod=n,
        boxes=[Word('What', Ty('n', Ob('n', z=-2), Ob('s', z=-1))),
               Word('Alice', Ty('n')), Word('is', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
               Word('and', Ty('n', Ob('s', z=1), Ob('n', z=2), Ob('n', z=1),
                              Ob('n', z=-2), Ob('s', z=-1), 'n')),
               Word('is', Ty(Ob('n', z=1), 's', Ob('n', z=-1))),
               Word('not', Ty(Ob('n', z=2), Ob('n', z=1), 's', Ob('n', z=-1))),
               Cup(Ty(Ob('n', z=-1)), Ty('n')), Cup(Ty('s'), Ty(Ob('s', z=1))),
               Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))), Cup(Ty('n'), Ty(Ob('n', z=1))),
               Cup(Ty(Ob('s', z=-1)), Ty('s')), Cup(Ty(Ob('n', z=-2)), Ty(Ob('n', z=-1))),
               Cup(Ty(Ob('n', z=1)), Ty(Ob('n', z=2))), Cup(Ty('n'), Ty(Ob('n', z=1))),
               Cup(Ty(Ob('s', z=-1)), Ty('s')), Cup(Ty(Ob('n', z=-2)), Ty(Ob('n', z=-1)))],
        offsets=[0, 3, 4, 7, 14, 17, 6, 5, 4, 7, 6, 5, 4, 3, 2, 1]
    )

    assert remove_swaps(inp_diagr) == out_diagr


def test_unknown_words_rewrite_rule():
    diagram = (Word('Alice', N) @ Word('loves', N >> S << N) @ Word('Bob', N)
               >> Cup(N, N.r) @ Id(S) @ Cup(N.l, N))

    vocab = ['Alice', 'Bob']
    rule = UnknownWordsRewriteRule(vocabulary=vocab)

    rewriter = Rewriter([rule])
    rewritten_diagram = rewriter(diagram)

    expected_diagram = (Word('Alice', N) @ Word('<UNK>', N >> S << N)
                        @ Word('Bob', N)
                        >> Cup(N, N.r) @ Id(S) @ Cup(N.l, N))

    assert rewritten_diagram == expected_diagram


@pytest.mark.parametrize('ignore_types', [False, True])
def test_handle_unknown_words(ignore_types):
    train_diagram = (Word('Alice', N) @ Word('loves', N >> S << N)
                     @ Word('Bob', N)
                     >> Cup(N, N.r) @ Id(S) @ Cup(N.l, N))

    rewrite_unknown_words = Rewriter([
        UnknownWordsRewriteRule.from_diagrams(diagrams=[train_diagram],
                                              min_freq=1,
                                              ignore_types=ignore_types)
    ])
    if ignore_types:
        assert rewrite_unknown_words.rules[0].vocabulary == {'Alice', 'loves', 'Bob'}
    else:
        assert rewrite_unknown_words.rules[0].vocabulary == {('Alice', N),
                                                             ('loves', N >> S << N),
                                                             ('Bob', N)}

    processed_train_diagram = rewrite_unknown_words(train_diagram)
    assert processed_train_diagram == train_diagram

    test_diagram = (Word('Alice', N) @ Word('loves', N >> S << N)
                    @ Word('Charlie', N)
                    >> Cup(N, N.r) @ Id(S) @ Cup(N.l, N))
    expected_test_diagram = (Word('Alice', N) @ Word('loves', N >> S << N)
                             @ Word('<UNK>', N)
                             >> Cup(N, N.r) @ Id(S) @ Cup(N.l, N))
    processed_test_diagram = rewrite_unknown_words(test_diagram)
    assert processed_test_diagram == expected_test_diagram


def test_unknown_words_stairs():
    sentence = 'This is a sentence'
    diagram = stairs_reader.sentence2diagram(sentence)
    rewrite_unknown_words = Rewriter([
        UnknownWordsRewriteRule.from_diagrams([diagram])
    ])

    assert rewrite_unknown_words(diagram) == diagram
