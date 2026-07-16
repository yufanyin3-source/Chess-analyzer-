"""chess_analyzer.pipeline - 特征提取 / 推理 / 报告 三条端到端流水线。"""
from chess_analyzer.pipeline.extract import run_feature_pipeline
from chess_analyzer.pipeline.predict import run_inference
from chess_analyzer.pipeline.report import run_style_report

__all__ = ["run_feature_pipeline", "run_inference", "run_style_report"]
