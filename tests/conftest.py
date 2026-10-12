import pytest

from simready_audit.audit import load_config


@pytest.fixture(scope="session")
def cfg():
    c, _ = load_config()
    return c
