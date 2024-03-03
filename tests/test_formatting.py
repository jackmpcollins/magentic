from typing_extensions import assert_type

from magentic.formatting import BulletedDict, BulletedList, NumberedDict, NumberedList


def test_bulleted_list():
    items = BulletedList(["foo", "bar", "baz"])
    assert_type(items, BulletedList[str])
    assert f"{items}" == "- foo\n- bar\n- baz"


def test_numbered_list():
    items = NumberedList(["foo", "bar", "baz"])
    assert_type(items, NumberedList[str])
    assert f"{items}" == "1. foo\n2. bar\n3. baz"


def test_bulleted_dict():
    items = BulletedDict({"foo": 1, "bar": 2, "baz": 3})
    assert_type(items, BulletedDict[str, int])
    assert f"{items}" == "- foo: 1\n- bar: 2\n- baz: 3"


def test_numbered_dict():
    items = NumberedDict({"foo": 1, "bar": 2, "baz": 3})
    assert_type(items, NumberedDict[str, int])
    assert f"{items}" == "1. foo: 1\n2. bar: 2\n3. baz: 3"
