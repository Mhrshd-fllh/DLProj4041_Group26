def test_imports():
    import src  # noqa: F401
    from src.utils.repro import set_seed  # noqa: F401
    from src.utils.run_logger import init_run, make_run_id  # noqa: F401

    assert callable(set_seed)
    assert callable(make_run_id)
    assert callable(init_run)
