from magentic.formatting import BulletedDict, BulletedList


def test_bulleted_list():
    items = BulletedList(["foo", "bar", "baz"])
    assert f"{items}" == "- foo\n- bar\n- baz"


def test_bulleted_dict():
    items = BulletedDict({"foo": 1, "bar": 2, "baz": 3})
    assert f"{items}" == "- foo: 1\n- bar: 2\n- baz: 3"
