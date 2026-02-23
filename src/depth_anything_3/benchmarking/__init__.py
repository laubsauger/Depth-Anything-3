# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmarking System for Depth Anything 3

Comprehensive benchmarking system for testing streaming performance
across different models, configurations, and hardware setups.
"""

from .config import (
    BenchmarkConfig,
    BenchmarkScenario,
    ModelConfig,
    StreamConfig,
    create_default_benchmark_config,
    PREDEFINED_MODELS,
)
from .metrics import BenchmarkMetrics, FrameMetrics, MetricsCollector
from .runner import BenchmarkRunner
from .report import ReportGenerator

__all__ = [
    "BenchmarkConfig",
    "BenchmarkScenario",
    "ModelConfig",
    "StreamConfig",
    "create_default_benchmark_config",
    "PREDEFINED_MODELS",
    "BenchmarkMetrics",
    "FrameMetrics",
    "MetricsCollector",
    "BenchmarkRunner",
    "ReportGenerator",
]
