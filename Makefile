.PHONY: install test smoke check analysis paper
install: ; pip install -e . && pip install -r requirements.txt
test:    ; python -m pytest -q tests
smoke:   ; python scripts/run_model.py --tier smoke --models seasonal_naive naive seasonal_avg drift --out results_smoke
check:   ; python scripts/check_models.py
paper:   ; cd paper && latexmk -pdf -interaction=nonstopmode main.tex
