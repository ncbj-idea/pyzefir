def assert_same_exception_list(
    actual_exception_list: list, exception_list: list
) -> None:
    assert len(actual_exception_list) == len(exception_list)

    actual_exception_list.sort(key=lambda x: str(x))
    exception_list.sort(key=lambda x: str(x))

    for actual, excepted in zip(actual_exception_list, exception_list):
        assert isinstance(actual, type(excepted))
        assert str(actual) == str(excepted)
