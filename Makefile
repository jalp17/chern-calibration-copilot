.PHONY: install test demo clean

install:
	pip install -r requirements.txt

test:
	python simulador/test_qsim.py

demo:
	streamlit run demo/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
