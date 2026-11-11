# Anchi DAC Bench — 26-Point AES17 & ITU-R Verification Suite
# Requires macOS 13.0+ on Apple Silicon (arm64) and Python 3.10+

DEFAULT_PYTHON := $(shell for p in python3.13 python3.12 python3.11 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do \
	if command -v "$$p" >/dev/null 2>&1 && "$$p" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then \
		command -v "$$p"; break; \
	fi; \
done)

PYTHON    ?= $(if $(DEFAULT_PYTHON),$(DEFAULT_PYTHON),python3)
VENV_DIR  := .venv
VENV_PY   := $(VENV_DIR)/bin/python
DYLIB     ?= lib/libNaitDSPBench.dylib
DAY       ?= 1

.PHONY: all test plots audition interactive venv clean distclean

all: test

venv:
	@if [ ! -x "$(VENV_PY)" ]; then \
		echo "==> Creating virtual environment in $(VENV_DIR) using $(PYTHON) and installing requirements.txt..."; \
		"$(PYTHON)" -m venv "$(VENV_DIR)"; \
		"$(VENV_DIR)/bin/pip" install --upgrade pip -q; \
		"$(VENV_DIR)/bin/pip" install -r requirements.txt; \
	fi

test:
	@if [ -x "$(VENV_PY)" ]; then \
		MPLCONFIGDIR=/tmp/anchi-mpl-cache "$(VENV_PY)" scripts/audio_loopback_test.py --all --no-audition-wavs --dylib "$(DYLIB)"; \
	elif "$(PYTHON)" -c "import numpy, scipy, matplotlib" >/dev/null 2>&1; then \
		MPLCONFIGDIR=/tmp/anchi-mpl-cache "$(PYTHON)" scripts/audio_loopback_test.py --all --no-audition-wavs --dylib "$(DYLIB)"; \
	else \
		$(MAKE) venv; \
		MPLCONFIGDIR=/tmp/anchi-mpl-cache "$(VENV_PY)" scripts/audio_loopback_test.py --all --no-audition-wavs --dylib "$(DYLIB)"; \
	fi

plots: test

audition:
	@if [ -x "$(VENV_PY)" ]; then \
		MPLCONFIGDIR=/tmp/anchi-mpl-cache "$(VENV_PY)" scripts/audio_loopback_test.py --all --dylib "$(DYLIB)"; \
	elif "$(PYTHON)" -c "import numpy, scipy, matplotlib" >/dev/null 2>&1; then \
		MPLCONFIGDIR=/tmp/anchi-mpl-cache "$(PYTHON)" scripts/audio_loopback_test.py --all --dylib "$(DYLIB)"; \
	else \
		$(MAKE) venv; \
		MPLCONFIGDIR=/tmp/anchi-mpl-cache "$(VENV_PY)" scripts/audio_loopback_test.py --all --dylib "$(DYLIB)"; \
	fi

interactive:
	@if [ -x "$(VENV_PY)" ]; then \
		MPLCONFIGDIR=/tmp/anchi-mpl-cache "$(VENV_PY)" scripts/audio_loopback_test.py --day "$(DAY)" --interactive --audition-target local --dylib "$(DYLIB)"; \
	elif "$(PYTHON)" -c "import numpy, scipy, matplotlib" >/dev/null 2>&1; then \
		MPLCONFIGDIR=/tmp/anchi-mpl-cache "$(PYTHON)" scripts/audio_loopback_test.py --day "$(DAY)" --interactive --audition-target local --dylib "$(DYLIB)"; \
	else \
		$(MAKE) venv; \
		MPLCONFIGDIR=/tmp/anchi-mpl-cache "$(VENV_PY)" scripts/audio_loopback_test.py --day "$(DAY)" --interactive --audition-target local --dylib "$(DYLIB)"; \
	fi

clean:
	rm -rf reports/loopback/audition_clips reports/loopback/listening-notes.md .mplconfig scripts/__pycache__ scripts/loopback_lab/__pycache__

distclean: clean
	rm -rf "$(VENV_DIR)"
