import pytest

from discoket.ccg2discocat.ccgbank_parser import CCGBankParseError, CCGBankParser


@pytest.fixture
def minimal_ccgbank(tmp_path_factory):
    root = tmp_path_factory.mktemp('ccgbank')
    auto_directory = root / 'data' / 'AUTO'
    auto_directory.mkdir(parents=True)

    good_directory = auto_directory / '00'
    good_directory.mkdir()
    good_file = good_directory / 'wsj_0001.auto'
    good_file.write_text(r'''ID=wsj_0001.1 PARSER=GOLD NUMPARSE=1
(<T S[dcl] 0 2> (<T S[dcl] 1 2> (<T NP 0 2> (<T NP 0 2> (<T NP 0 2> (<T NP 0 1> (<T N 1 2> (<L N/N NNP NNP Pierre N_73/N_73>) (<L N NNP NNP Vinken N>) ) ) (<L , , , , ,>) ) (<T NP\NP 0 1> (<T S[adj]\NP 1 2> (<T NP 0 1> (<T N 1 2> (<L N/N CD CD 61 N_93/N_93>) (<L N NNS NNS years N>) ) ) (<L (S[adj]\NP)\NP JJ JJ old (S[adj]\NP_83)\NP_84>) ) ) ) (<L , , , , ,>) ) (<T S[dcl]\NP 0 2> (<L (S[dcl]\NP)/(S[b]\NP) MD MD will (S[dcl]\NP_10)/(S[b]_11\NP_10:B)_11>) (<T S[b]\NP 0 2> (<T S[b]\NP 0 2> (<T (S[b]\NP)/PP 0 2> (<L ((S[b]\NP)/PP)/NP VB VB join ((S[b]\NP_20)/PP_21)/NP_22>) (<T NP 1 2> (<L NP[nb]/N DT DT the NP[nb]_29/N_29>) (<L N NN NN board N>) ) ) (<T PP 0 2> (<L PP/NP IN IN as PP/NP_34>) (<T NP 1 2> (<L NP[nb]/N DT DT a NP[nb]_48/N_48>) (<T N 1 2> (<L N/N JJ JJ nonexecutive N_43/N_43>) (<L N NN NN director N>) ) ) ) ) (<T (S\NP)\(S\NP) 0 2> (<L ((S\NP)\(S\NP))/N[num] NNP NNP Nov. ((S_61\NP_56)_61\(S_61\NP_56)_61)/N[num]_62>) (<L N[num] CD CD 29 N[num]>) ) ) ) ) (<L . . . . .>) )
ID=wsj_0001.2 PARSER=GOLD NUMPARSE=1
(<T S[dcl] 0 2> (<T S[dcl] 1 2> (<T NP 0 1> (<T N 1 2> (<L N/N NNP NNP Mr. N_142/N_142>) (<L N NNP NNP Vinken N>) ) ) (<T S[dcl]\NP 0 2> (<L (S[dcl]\NP)/NP VBZ VBZ is (S[dcl]\NP_87)/NP_88>) (<T NP 0 2> (<T NP 0 1> (<L N NN NN chairman N>) ) (<T NP\NP 0 2> (<L (NP\NP)/NP IN IN of (NP_99\NP_99)/NP_100>) (<T NP 0 2> (<T NP 0 1> (<T N 1 2> (<L N/N NNP NNP Elsevier N_109/N_109>) (<L N NNP NNP N.V. N>) ) ) (<T NP[conj] 1 2> (<L , , , , ,>) (<T NP 1 2> (<L NP[nb]/N DT DT the NP[nb]_131/N_131>) (<T N 1 2> (<L N/N NNP NNP Dutch N_126/N_126>) (<T N 1 2> (<L N/N VBG VBG publishing N_119/N_119>) (<L N NN NN group N>) ) ) ) ) ) ) ) ) ) (<L . . . . .>) )''')

    bad_directory = auto_directory / '25'
    bad_directory.mkdir()
    bad_file = bad_directory / 'wsj_2500.auto'
    bad_file.write_text('''ID=wsj_2501.1 PARSER=GOLD NUMPARSE=1
Bad tree line
Bad ID line''')

    return root


def test_ccgbank_parser(minimal_ccgbank):
    ccgbank_parser = CCGBankParser(minimal_ccgbank)
    good_diagrams = ccgbank_parser.section2diagrams(0)
    assert len(good_diagrams) == 2 and all(good_diagrams)

    bad_diagrams = ccgbank_parser.section2diagrams(25, suppress_exceptions=True)
    assert list(bad_diagrams.values()) == [None]
    with pytest.raises(CCGBankParseError):
        ccgbank_parser.section2diagrams(25)