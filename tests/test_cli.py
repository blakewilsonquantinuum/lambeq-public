import pytest
import argparse

from lambeq import cli
from lambeq.cli import ArgumentList


@pytest.fixture
def arg_parser() -> argparse.ArgumentParser:
    return cli.prepare_parser()


def test_args_validation_two_inputs(arg_parser):
    cli_args = arg_parser.parse_args(["--input_file", "dummy.txt",
                                      "Input sentence."])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_missing_output_file(arg_parser):
    cli_args = arg_parser.parse_args(["--output_format", "pickle",
                                      "Input sentence."])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_both_parser_and_reader_given(arg_parser):
    cli_args = arg_parser.parse_args(["--parser", "depccg",
                                      "--reader", "spiders",
                                      "Input sentence."])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_redundant_output_args(arg_parser):
    cli_args = arg_parser.parse_args(["--output_format", "text-unicode",
                                      "--output_options", "font_size=12",
                                      "-t", "Input sentence."])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_invalid_output_dir(arg_parser):
    cli_args = arg_parser.parse_args(["--output_format", "text-unicode",
                                      "--output_dir", "dummy_folder",
                                      "Input sentence."])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_invalid_output_format_ansatz(arg_parser):
    cli_args = arg_parser.parse_args(["--output_format", "text-unicode",
                                      "--output_file", "dummy_file",
                                      "--ansatz", "iqp",
                                      "Input sentence."])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_invalid_output_format_rewrite(arg_parser):
    cli_args = arg_parser.parse_args(["--output_format", "text-ascii",
                                      "--rewrite_rules", "determiner",
                                      "--output_file", "dummy_file",
                                      "Input sentence."])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_args_validation_invalid_output_format_reader(arg_parser):
    cli_args = arg_parser.parse_args(["--output_format", "text-ascii",
                                      "--output_file", "dummy_file",
                                      "--reader", "spiders",
                                      "Input sentence."])
    with pytest.raises(ValueError):
        cli.validate_args(cli_args)


def test_correct_arglist_parsing(arg_parser):
    cli_args = arg_parser.parse_args(["--output_format", "image",
                                      "--output_options", "fig_width=15",
                                      "fig_height=5", "font_size=3",
                                      "--output_file", "dummy_file",
                                      "Input sentence."])
    assert cli_args.output_options["fig_width"] == 15
    assert cli_args.output_options["fig_height"] == 5
    assert cli_args.output_options["font_size"] == 3


@pytest.fixture
def arg_list() -> ArgumentList:
    return ArgumentList([("test_int", int, 4), ("test_str", str, None)])


def test_arglist_all_options(arg_list):
    assert arg_list.all_options() == ("test_int=<int> (default: 4), "
                                      "test_str=<str> (default: None)")


def test_arglist_generator(arg_list):
    all_options = ", ".join(x for x in arg_list)
    assert all_options == arg_list.all_options()


def test_type_mismatch():
    with pytest.raises(ValueError):
        _ = ArgumentList([["test_int", int, 4.3], ["test_str", str, 3]])


def test_access(arg_list):
    assert "test_int=5" in arg_list
    assert "test_str=text" in arg_list
    assert "test_int=text" not in arg_list
    assert "test_bool=True" not in arg_list
    assert arg_list("test_int=6") == ("test_int", 6)
