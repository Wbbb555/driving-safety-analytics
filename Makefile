# 一键复现：make all
# Windows 上若没有 make，直接照 README 里的命令逐条跑。

PY ?= python

.PHONY: all data traps task1 task2 test claims figs clean

all: data traps task1 task2 test claims

data:
	$(PY) scripts/generate_synthetic.py

traps:
	$(PY) scripts/demo_traps.py --json outputs/metrics.json

task1:
	$(PY) src/task1/run_task1.py

task2:
	$(PY) src/task2/run_task2.py

test:
	$(PY) -m pytest tests -q

claims:
	$(PY) scripts/check_claims.py

figs:
	$(PY) scripts/make_figures.py

# 清空生成物，用来验证「从零可复现」。
clean:
	rm -rf data/synthetic outputs
