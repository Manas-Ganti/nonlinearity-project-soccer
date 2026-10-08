PY ?= .venv/bin/python
CLI = $(PY) -m src.cli.main

.PHONY: help venv test lint fmt check clean \
        ingest-understat ingest-statsbomb build-understat build-statsbomb build-synthetic \
        calibrate-dedup fit-null recovery power rounding-bias fit-dev fit-holdout \
        estimator-comparison tau-profile cross-source secondary report notebook demo pipeline \
        build-possessions poss-recovery poss-power poss-fit

help:  ## list targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

venv:  ## create .venv and install everything
	python3 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -e ".[dev]"

test:  ## the fast suite (no network, no overnight data)
	$(PY) -m pytest -q

lint:  ## ruff
	.venv/bin/ruff check .

fmt:  ## ruff format
	.venv/bin/ruff format .

check: lint test  ## what must pass before anything ships

# ---------------------------------------------------------------- build order

ingest-understat:  ## step 1: ~23,000 requests. Run once, overnight. Never re-scrape.
	$(CLI) -v ingest-understat

ingest-statsbomb:  ## step 1: the second-resolution sample (minutes, not hours)
	$(CLI) -v ingest-statsbomb

build-understat:  ## step 1: raw -> canonical event table
	$(CLI) -v build --source understat

build-statsbomb:
	$(CLI) -v build --source statsbomb

ingest-wyscout:  ## step 1: extract the figshare release already in data/raw/wyscout
	$(CLI) -v ingest-wyscout

build-wyscout:
	$(CLI) -v build --source wyscout

build-possessions:  ## richer events: Wyscout possession table (docs/spec_possessions.md)
	$(CLI) -v build-possessions

poss-recovery:  ## possessions model A, gate 2 (A GATE: exits 2 on failure)
	$(CLI) -v poss-recovery

poss-power:  ## possessions model A, gate 3: detection floor
	$(CLI) -v poss-power

poss-fit:  ## possessions model A, gate 4: development fit (exploratory)
	$(CLI) -v poss-fit

build-pooled: build-statsbomb build-wyscout  ## step 1: the pooled second-resolution slate (the headline)
	$(CLI) -v build --source pooled

build-synthetic:  ## a slate with known ground truth, for running the pipeline with no data
	$(CLI) -v build --source synthetic --n-matches 800

calibrate-dedup:  ## step 2: where mechanical follow-ups end, per provider's second clock
	$(CLI) -v calibrate-dedup --source statsbomb
	$(CLI) -v calibrate-dedup --source wyscout

fit-null:  ## step 3: the background model, checked by time rescaling
	$(CLI) -v fit-null --source $(or $(SOURCE),pooled)

recovery:  ## step 5: plant eta=0.3 and find it. THIS IS A GATE.
	$(CLI) -v recovery --source $(or $(SOURCE),pooled) --replicates $(or $(R),10)

power:  ## step 6: the detection floor
	$(CLI) -v power --source $(or $(SOURCE),pooled) --replicates $(or $(R),30)

rounding-bias:  ## step 7: second vs minute resolution, on the same second-clock events
	$(CLI) -v rounding-bias --source $(or $(SOURCE),pooled)

estimator-comparison:  ## why the joint fit is the default and two-stage is not
	$(CLI) -v estimator-comparison --source $(or $(SOURCE),pooled)

tau-profile:  ## is the kernel timescale identified at all?
	$(CLI) -v tau-profile --source $(or $(SOURCE),pooled)

cross-source:  ## Understat vs StatsBomb on the fixtures both cover
	$(CLI) -v cross-source

secondary:  ## extensions: chance quality, state-dependent eta
	$(CLI) -v secondary --source $(or $(SOURCE),pooled)

# fit-dev reads the detection floor from results/power_<source>_<split>.json, so `power`
# must have been run for the same slate first. It says so loudly if it has not.
fit-dev:  ## step 8: the headline fit on the development seasons
	$(CLI) -v fit --source $(or $(SOURCE),pooled) --split dev \
		--bootstrap $(or $(B),500) --cluster-bootstrap $(or $(CB),200)

fit-holdout:  ## step 9: run once, report, stop.
	$(CLI) -v fit --source $(or $(SOURCE),pooled) --split holdout --final-run \
		--bootstrap $(or $(B),500) --cluster-bootstrap $(or $(CB),200)

report:  ## assemble results/ into docs/RESULTS.md
	$(PY) -m src.cli.report

notebook:  ## open the results notebook (needs: pip install -e ".[viz]")
	.venv/bin/jupyter lab notebooks/results.ipynb

pipeline: calibrate-dedup fit-null estimator-comparison recovery power tau-profile rounding-bias fit-dev report  ## everything but the holdout

demo:  ## end-to-end on synthetic data: no network, no overnight pull, a few minutes
	$(CLI) -v build --source synthetic --n-matches 600
	$(CLI) -v fit-null --source synthetic --split all --final-run
	$(CLI) -v recovery --source synthetic --split all --final-run --replicates 5
	$(CLI) -v power --source synthetic --split all --final-run --replicates 8 --null-replicates 30

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
