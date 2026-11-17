"""视觉记忆植入（Visual Inception）专用类别多样化检索评估入口。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.visual_inception_category_evaluation import main

if __name__ == '__main__':
    raise SystemExit(main())
