import pytest

from precision_planner.formats import Format, Rounding


@pytest.mark.parametrize(
    "fmt,expected",
    [(Format.FP32, 4), (Format.FP16, 2), (Format.INT8, 1), (Format.FP8, 1)],
)
def test_bytes_per_param(fmt, expected):
    assert fmt.bytes_per_param == expected


@pytest.mark.parametrize("fmt", list(Format))
def test_label_is_uppercase_value(fmt):
    assert fmt.label == fmt.value.upper()


def test_label_example():
    assert Format.FP16.label == "FP16"


def test_format_equals_plain_string():
    assert "fp16" == Format.FP16
    assert Format("int8") is Format.INT8


def test_format_rejects_unknown_value():
    with pytest.raises(ValueError):
        Format("bf16")


def test_rounding_from_string():
    assert Rounding("stochastic") is Rounding.STOCHASTIC
    assert Rounding("nearest") == "nearest"


def test_rounding_rejects_unknown_value():
    with pytest.raises(ValueError):
        Rounding("floor")
