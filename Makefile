DATASET ?= bottles-and-containers
RUN = uv run simready-audit --dataset $(DATASET)

.PHONY: all setup download test audit fix mutate controls verify report

all: setup download test audit fix mutate controls verify report

setup:
	uv sync --frozen

download:
	$(RUN) download
	uv run simready-audit --dataset yam-authored download
	uv run simready-audit --dataset yam-default download

test:
	uv run pytest -q

audit:
	$(RUN) audit

fix:
	$(RUN) fix

mutate:
	$(RUN) mutate

controls:
	$(RUN) controls

verify:
	$(RUN) verify

report:
	$(RUN) report
