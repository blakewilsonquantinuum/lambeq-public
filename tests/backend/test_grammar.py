from pytest import raises

from lambeq.backend.grammar import *

def test_Ty():
    a , b  = map(Ty, 'ab')
    c = Ty()
    ab = a @ b

    assert a.r.l == a == a.l.r
    assert a.is_adjoint(a.r) and a.r.is_adjoint(a)
    assert a.is_atomic and b.is_atomic
    assert ab.is_complex
    assert c.is_empty
    assert a.l.z == -1
    assert len(a) == 1 and len(ab) == 2
    assert list(ab) == [a, b]
    assert ab[0] == a and ab[1] == b and a[0] == a
    assert a @ c == a
    assert a.wind(0) == a
    assert a.wind(42).unwind() == a
    assert a << b == a @ b.l
    assert a >> b == a.r @ b
    assert a**0 == Ty()
    assert a**1 == a
    assert a**2 == a @ a
    assert a**3 == a @ a @ a

    with raises(TypeError):
        a @ 'b'
    with raises(TypeError):
        a << 'b'
    with raises(TypeError):
        a >> 'b'

def test_Cup_init():
    t = Ty('n')
    with raises(ValueError):
        Cup(t@t, (t@t).l)
    with raises(ValueError):
        Cup(Ty(), Ty())

def test_Box():
    a , b  = map(Ty, 'ab')
    A = Box('richie', a, b)

    assert A.dagger().dagger() == A
    assert A.name == 'richie' and A.dom == a and A.cod == b
    assert A.r.l == A and A.l.r == A
    assert A.wind(42).z == 42
    assert A.wind(42).unwind() == A
    assert A.dagger() >> Id(a)

def test_Box_magics():
    a , b  = map(Ty, 'ab')
    A, B = Box('ian', a, b), Box('charlie', b, a)

    assert A.to_diagram() == Diagram(A.dom, A.cod, layers=[
        Layer(box=A, left=Ty(), right=Ty())])

    # Tensoring Boxes/ Diagrams --> Diagram
    assert A @ B == Diagram(a@b, b@a, layers=[
        Layer(box=A, left=Ty(), right=b),
        Layer(box=B, left=b, right=Ty())])
    assert A @ B.to_diagram() == Diagram(a@b, b@a, layers=[
        Layer(box=A, left=Ty(), right=b),
        Layer(box=B, left=b, right=Ty())])

    # Concatenating Boxes/ Diagrams --> Diagram
    assert A >> B == Diagram(a, a, layers=[
        Layer(box=A, left=Ty(), right=Ty()),
        Layer(box=B, left=Ty(), right=Ty())])
    assert A >> B.to_diagram() == Diagram(a, a, layers=[
        Layer(box=A, left=Ty(), right=Ty()),
        Layer(box=B, left=Ty(), right=Ty())])

def test_Word():
    n = Ty('n')
    word = Word('bob', n)

    assert word.dom == Ty()
    assert word.z == 0
    assert word.dagger().dagger() == word
    assert word.l.r == word.r.l == word
    assert word.dagger().r.dagger().l == word == word.l.dagger().r.dagger()

def test_pregroup():
    s, n, x = Ty('s'), Ty('n'), Ty('x')
    Alice, Bob = Word("Alice", n), Word("Bob", n)
    loves = Word('loves', n.r @ s @ n.l)
    sentence = Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)

    cup, cap = Cup(n, n.r), Cap(n.r, n)
    f, g, h = Box('f', n, n), Box('g', s @ n, n), Box('h', n, n @ s)
    diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h

    assert sentence.is_pregroup()
    assert not diagram.is_pregroup()


def test_Cup():
    a, b = map(Ty, 'ab')
    ab = a @ b

    # Errors when instantiating
    with raises(ValueError):
        Cup(ab, ab.r)
    with raises(ValueError):
        Cup(a, a)

    cup = Cup(a, a.r)
    assert cup.l.r == cup
    assert cup.dagger().dagger() == cup

def test_Cap():
    a, b = map(Ty, 'ab')
    ab = a @ b

    # Errors when instantiating
    with raises(ValueError):
        Cap(ab, ab.r)
    with raises(ValueError):
        Cap(a, a)

    cap = Cap(a, a.r)
    assert cap.l.r == cap
    assert cap.dagger().dagger() == cap

def test_Spider():
    a, b = map(Ty, 'ab')
    ab = a @ b

    with raises(TypeError):
        Spider(ab, 2, 2)

    spider = Spider(a, 2, 2)
    assert spider.r.l == spider
    assert spider.dagger().dagger() == spider

def test_Swap():
    a, b = map(Ty, 'ab')
    ab = a @ b

    with raises(ValueError):
        Swap(ab, a)

    swap = Swap(a, b)
    assert swap.dagger().dagger() == swap
    assert swap.l.r == swap

def test_Layer():
    a, b, c, d = map(Ty, 'abcd')
    box = Box('nikhil', a, b)

    layer = Layer(box=box, left=Ty(), right=Ty())

    assert layer.extend() == layer
    assert layer.extend(left=c, right=d) == Layer(box=box, left=c, right=d)

def test_Id():
    a, b = map(Ty, 'ab')
    box = Box('dimitri', a, b)

    assert box @ Id() == box.to_diagram()
    assert box @ Id(b) >> Id(b) @ box.dagger() == box @ box.dagger()
    assert Id().is_id == True
    assert Id().dagger() == Id()

def test_Diagram():
    n, s = Ty('n'), Ty('s')
    cup, cap = Cup(n, n.r), Cap(n.r, n)
    f, g, h = Box('f', n, n), Box('g', s @ n, n), Box('h', n, n @ s)
    diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h

    assert diagram.boxes == [g, cap, f.dagger(), f, cup, h]
    assert diagram.offsets == [0, 1, 0, 2, 0, 0]
    assert diagram.l.r == diagram
    assert diagram.dagger().dagger() == diagram
    assert diagram.is_pregroup() == False

def test_Pregroup_Diagram():
    n, s = Ty('n'), Ty('s')
    words = [Word('she', n), Word('goes', n.r @ s @ n.l), Word('home', n)]
    morphisms = [(Cup, 0, 1), (Cup, 3, 4)]
    diagram = Diagram.create_pregroup_diagram(words, morphisms)
    assert diagram == words[0] @ words[1] @ words[2] >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)

def test_Diagram_NotImplemented():

    class Dummy:
        def to_diagram(self):
            return self

    n, s = Ty('n'), Ty('s')
    cup, cap = Cup(n, n.r), Cap(n.r, n)
    f, g, h = Box('f', n, n), Box('g', s @ n, n), Box('h', n, n @ s)
    diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h

    dummy_diagram = Dummy()

    with raises(TypeError):
        diagram @ 'something very wrong'
    with raises(TypeError):
        diagram @ dummy_diagram
    with raises(TypeError):
        diagram >> 'something very wrong'
    with raises(TypeError):
        diagram >> dummy_diagram
    with raises(ValueError):
        diagram >> Box('thomas', Ty('something wrong'), Ty())

def test_Dagger():
    n, s = Ty('n'), Ty('s')
    box = Box('bob', n, s)

    assert box.l.dagger().r.dagger() == box.dagger().l.dagger().r == box
