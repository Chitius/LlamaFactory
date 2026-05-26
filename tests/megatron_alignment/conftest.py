import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-l5",
        action="store_true",
        default=False,
        help="Run the L5 loss alignment test (requires GPU + Megatron).",
    )
