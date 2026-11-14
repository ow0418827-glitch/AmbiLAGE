"""从自包含冻结基础包串行生成、检索、复算并报告。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.base_workflow import main

if __name__ == '__main__':
    raise SystemExit(main())
