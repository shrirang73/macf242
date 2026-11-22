#!/usr/bin/env bash
# Full reproduction of Experiment B, from the raw bhavcopy to the report PDFs.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
step() { echo; echo "=== $1  ($(date +%H:%M:%S))"; }
step tests;        $PY -m pytest tests -q
step chains;       $PY -m heston.data
step calibration;  $PY -u -m heston.run_calibration --workers 5
step oos;          $PY -m heston.oos
step hedge;        $PY -m heston.hedge
step paired;       $PY -m heston.analyse_hedge
step simulation;   $PY -u -m heston.simulate
step assets;       $PY -m heston.report_assets
step demo;         $PY -m heston.demo_data
step latex;        cd report && for f in final_report report; do pdflatex -interaction=nonstopmode $f >/dev/null; pdflatex -interaction=nonstopmode $f | grep "Output written"; done
step done
