PY := .venv/bin/python

.PHONY: setup all test clean churn risk serve feedback monitor

setup:            ## create .venv and install pinned dependencies
	./setup.sh

all:              ## tests, then every pipeline, from a clean slate
	$(PY) run_all.py

test:             ## just the test suite
	$(PY) -m pytest tests/ -q

churn:    ; $(PY) 01_churn_batch_pipeline.py
risk:     ; $(PY) 02_risk_realtime_pipeline.py
serve:    ; $(PY) 03_live_scoring_demo.py
feedback: ; $(PY) 04_feedback_loops.py
monitor:  ; $(PY) 05_monitoring_two_clocks.py

clean:            ## remove every generated artifact
	rm -rf outputs
