.PHONY: install data centralized federated evaluate compare app test clean

install:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

data:
	. venv/bin/activate && python scripts/download_data.py

centralized:
	. venv/bin/activate && python scripts/train_centralized.py

federated:
	. venv/bin/activate && python scripts/train_federated.py

evaluate:
	. venv/bin/activate && python scripts/evaluate.py --model outputs/models/federated_global_best.pt

compare:
	. venv/bin/activate && python scripts/compare.py

app:
	. venv/bin/activate && streamlit run scripts/app.py

test:
	. venv/bin/activate && python tests/test_smoke.py

clean:
	find . -type d -name __pycache__ -not -path './venv/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
