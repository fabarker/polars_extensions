SHELL=/bin/bash

.venv:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip setuptools wheel maturin
	.venv/bin/pip install -r requirements.txt

install: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop

install-release: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release

pre-commit: .venv
	cargo +nightly fmt --all && cargo clippy --all-features
	.venv/bin/python -m ruff check . --fix --exit-non-zero-on-fix
	.venv/bin/python -m ruff format polars_extensions tests
	.venv/bin/mypy polars_extensions tests

test: .venv
	.venv/bin/python -m pytest tests

run: install
	source .venv/bin/activate && python run.py

run-release: install-release
	source .venv/bin/activate && python run.py

setup-nightly:
	rustup toolchain install nightly
	rustup component add rustfmt --toolchain nightly
	rustup component add clippy --toolchain nightly
