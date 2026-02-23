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
Report Generation

Creates HTML reports with charts and visualizations for benchmark results.
"""

from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

from .metrics import BenchmarkMetrics


class ReportGenerator:
    """
    Generates HTML reports from benchmark results.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: List[BenchmarkMetrics],
        benchmark_name: str,
        description: str = "",
    ) -> Path:
        """
        Generate HTML report from benchmark results.

        Args:
            results: List of benchmark metrics
            benchmark_name: Name of the benchmark
            description: Optional description

        Returns:
            Path to generated HTML report
        """
        report_path = self.output_dir / "report.html"

        # Generate HTML
        html = self._generate_html(results, benchmark_name, description)

        # Write to file
        with open(report_path, 'w') as f:
            f.write(html)

        print(f"📄 Generated HTML report: {report_path}")
        return report_path

    def _generate_html(
        self,
        results: List[BenchmarkMetrics],
        benchmark_name: str,
        description: str,
    ) -> str:
        """Generate complete HTML report."""
        # Prepare data for charts
        chart_data = self._prepare_chart_data(results)

        # Generate HTML sections
        header = self._generate_header(benchmark_name, description)
        summary = self._generate_summary_table(results)
        comparison = self._generate_comparison_charts(chart_data)
        timing_breakdown = self._generate_timing_breakdown_section(chart_data)
        charts = self._generate_charts(chart_data)
        detailed = self._generate_detailed_results(results)

        # Combine into full HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{benchmark_name} - Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    {header}
    <div class="container">
        {summary}
        {comparison}
        {timing_breakdown}
        {charts}
        {detailed}
    </div>
    <script>
        {self._generate_chart_scripts(chart_data)}
    </script>
</body>
</html>
"""
        return html

    def _get_css(self) -> str:
        """Get CSS styles for the report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            line-height: 1.6;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .header .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .header .timestamp {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 0.5rem;
        }

        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
        }

        .section {
            background: white;
            border-radius: 8px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .section h2 {
            color: #667eea;
            margin-bottom: 1.5rem;
            font-size: 1.8rem;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }

        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #e1e8ed;
        }

        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }

        tr:hover {
            background: #f8f9fa;
        }

        .metric-value {
            font-weight: 600;
            color: #667eea;
        }

        .metric-good {
            color: #28a745;
        }

        .metric-warning {
            color: #ffc107;
        }

        .metric-bad {
            color: #dc3545;
        }

        .chart-container {
            position: relative;
            height: 400px;
            margin: 2rem 0;
        }

        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }

        .scenario-details {
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: #f8f9fa;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }

        .scenario-details h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .stat-card {
            background: white;
            padding: 1rem;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .stat-label {
            font-size: 0.85rem;
            color: #6c757d;
            margin-bottom: 0.25rem;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #667eea;
        }

        .comparison-note {
            background: #e7f3ff;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }
        """

    def _generate_header(self, benchmark_name: str, description: str) -> str:
        """Generate report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
        <div class="header">
            <h1>📊 {benchmark_name}</h1>
            <div class="subtitle">{description}</div>
            <div class="timestamp">Generated: {timestamp}</div>
        </div>
        """

    def _generate_summary_table(self, results: List[BenchmarkMetrics]) -> str:
        """Generate summary table comparing all scenarios."""
        rows = []
        for metric in results:
            rows.append(f"""
            <tr>
                <td><strong>{metric.scenario_name}</strong></td>
                <td class="metric-value">{metric.avg_fps:.2f}</td>
                <td>{metric.avg_latency_ms:.2f}</td>
                <td>{metric.p95_latency_ms:.2f}</td>
                <td>{metric.avg_memory_mb:.2f}</td>
                <td>{metric.max_memory_mb:.2f}</td>
            </tr>
            """)

        return f"""
        <div class="section">
            <h2>📈 Performance Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Scenario</th>
                        <th>Avg FPS</th>
                        <th>Avg Latency (ms)</th>
                        <th>P95 Latency (ms)</th>
                        <th>Avg Memory (MB)</th>
                        <th>Peak Memory (MB)</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _prepare_chart_data(self, results: List[BenchmarkMetrics]) -> Dict[str, Any]:
        """Prepare data for charts."""
        scenario_names = [r.scenario_name for r in results]

        # Prepare timing breakdown data
        timing_data = {}
        for r in results:
            if r.timing_stats and r.timing_stats.mean_breakdown:
                timing_data[r.scenario_name] = r.timing_stats.mean_breakdown

        return {
            "scenarios": scenario_names,
            "fps": [r.avg_fps for r in results],
            "latency_avg": [r.avg_latency_ms for r in results],
            "latency_p95": [r.p95_latency_ms for r in results],
            "memory_avg": [r.avg_memory_mb for r in results],
            "memory_max": [r.max_memory_mb for r in results],
            "temporal_jitter": [r.temporal_jitter for r in results],
            "temporal_smoothness": [r.temporal_smoothness for r in results],
            "flicker_score": [r.flicker_score for r in results],
            "frame_data": {
                r.scenario_name: [fm.total_time_ms for fm in r.frame_metrics]
                for r in results
            },
            "timing_breakdown": timing_data,
        }

    def _generate_timing_breakdown_section(self, data: Dict[str, Any]) -> str:
        """Generate detailed timing breakdown section."""
        timing_data = data.get("timing_breakdown", {})

        if not timing_data:
            return ""  # No timing data available

        return f"""
        <div class="section">
            <h2>⏱️ Detailed Timing Breakdown</h2>
            <div class="comparison-note">
                <strong>Note:</strong> This section shows where processing time is spent across different pipeline stages.
                Use this to identify bottlenecks and optimization opportunities.
            </div>
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="timingStackedChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="timingPercentageChart"></canvas>
                </div>
            </div>
        </div>
        """

    def _generate_comparison_charts(self, data: Dict[str, Any]) -> str:
        """Generate comparison charts showing all results at a glance."""
        return f"""
        <div class="section">
            <h2>🔍 At-a-Glance Comparison</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="comparisonFpsChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="comparisonTemporalChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="comparisonStabilityChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="comparisonLatencyMemoryChart"></canvas>
                </div>
            </div>
        </div>
        """

    def _generate_charts(self, data: Dict[str, Any]) -> str:
        """Generate chart containers."""
        return f"""
        <div class="section">
            <h2>📊 Performance Charts</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="fpsChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="latencyChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="memoryChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="frameTimeChart"></canvas>
                </div>
            </div>
        </div>
        """

    def _generate_chart_scripts(self, data: Dict[str, Any]) -> str:
        """Generate JavaScript for charts."""
        # Convert data to JSON
        scenarios_json = json.dumps(data["scenarios"])
        fps_json = json.dumps(data["fps"])
        latency_avg_json = json.dumps(data["latency_avg"])
        latency_p95_json = json.dumps(data["latency_p95"])
        memory_avg_json = json.dumps(data["memory_avg"])
        memory_max_json = json.dumps(data["memory_max"])
        temporal_jitter_json = json.dumps(data["temporal_jitter"])
        temporal_smoothness_json = json.dumps(data["temporal_smoothness"])
        flicker_score_json = json.dumps(data["flicker_score"])

        # Prepare frame data for all scenarios
        frame_data_dict = data["frame_data"]
        # Find the maximum number of frames across all scenarios
        max_frames = max(len(times) for times in frame_data_dict.values()) if frame_data_dict else 0
        frame_indices = list(range(max_frames))
        frame_indices_json = json.dumps(frame_indices)

        # Build datasets for all scenarios
        frame_datasets = []
        for idx, (scenario_name, frame_times) in enumerate(frame_data_dict.items()):
            color = f"rgba({102 + idx * 30 % 153}, {126 + idx * 40 % 129}, {234 - idx * 20 % 90}, 1)"
            frame_datasets.append({
                "label": scenario_name,
                "data": frame_times,
                "borderColor": color,
                "backgroundColor": color.replace("1)", "0.1)"),
                "borderWidth": 2,
                "pointRadius": 0.5,
            })
        frame_datasets_json = json.dumps(frame_datasets)

        # Prepare timing breakdown data
        timing_breakdown = data.get("timing_breakdown", {})
        has_timing = len(timing_breakdown) > 0

        timing_scripts = ""
        if has_timing:
            timing_scenarios = list(timing_breakdown.keys())
            timing_preprocessing = []
            timing_input_prep = []
            timing_model_forward = []
            timing_postprocessing = []
            timing_export = []

            for scenario in timing_scenarios:
                breakdown = timing_breakdown[scenario]
                timing_preprocessing.append(breakdown.total_preprocessing_ms)
                timing_input_prep.append(breakdown.total_input_prep_ms)
                timing_model_forward.append(breakdown.model_forward_ms)
                timing_postprocessing.append(breakdown.total_postprocessing_ms)
                timing_export.append(breakdown.export_ms)

            timing_scenarios_json = json.dumps(timing_scenarios)
            timing_preprocessing_json = json.dumps(timing_preprocessing)
            timing_input_prep_json = json.dumps(timing_input_prep)
            timing_model_forward_json = json.dumps(timing_model_forward)
            timing_postprocessing_json = json.dumps(timing_postprocessing)
            timing_export_json = json.dumps(timing_export)

            # Calculate percentages
            totals = [
                timing_preprocessing[i] + timing_input_prep[i] + timing_model_forward[i] +
                timing_postprocessing[i] + timing_export[i]
                for i in range(len(timing_scenarios))
            ]

            timing_preprocessing_pct = [(p / t * 100) if t > 0 else 0 for p, t in zip(timing_preprocessing, totals)]
            timing_input_prep_pct = [(p / t * 100) if t > 0 else 0 for p, t in zip(timing_input_prep, totals)]
            timing_model_forward_pct = [(p / t * 100) if t > 0 else 0 for p, t in zip(timing_model_forward, totals)]
            timing_postprocessing_pct = [(p / t * 100) if t > 0 else 0 for p, t in zip(timing_postprocessing, totals)]
            timing_export_pct = [(p / t * 100) if t > 0 else 0 for p, t in zip(timing_export, totals)]

            timing_preprocessing_pct_json = json.dumps(timing_preprocessing_pct)
            timing_input_prep_pct_json = json.dumps(timing_input_prep_pct)
            timing_model_forward_pct_json = json.dumps(timing_model_forward_pct)
            timing_postprocessing_pct_json = json.dumps(timing_postprocessing_pct)
            timing_export_pct_json = json.dumps(timing_export_pct)

            timing_scripts = f"""
        // Timing Breakdown Charts
        new Chart(document.getElementById('timingStackedChart'), {{
            type: 'bar',
            data: {{
                labels: {timing_scenarios_json},
                datasets: [
                    {{
                        label: 'Preprocessing',
                        data: {timing_preprocessing_json},
                        backgroundColor: 'rgba(255, 99, 132, 0.7)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Input Preparation',
                        data: {timing_input_prep_json},
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Model Forward',
                        data: {timing_model_forward_json},
                        backgroundColor: 'rgba(255, 206, 86, 0.7)',
                        borderColor: 'rgba(255, 206, 86, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Postprocessing',
                        data: {timing_postprocessing_json},
                        backgroundColor: 'rgba(75, 192, 192, 0.7)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Export',
                        data: {timing_export_json},
                        backgroundColor: 'rgba(153, 102, 255, 0.7)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Time Breakdown by Stage (Absolute)',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    tooltip: {{
                        callbacks: {{
                            footer: function(tooltipItems) {{
                                let sum = 0;
                                tooltipItems.forEach(function(tooltipItem) {{
                                    sum += tooltipItem.parsed.y;
                                }});
                                return 'Total: ' + sum.toFixed(2) + 'ms';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        stacked: true,
                        title: {{ display: true, text: 'Scenario' }}
                    }},
                    y: {{
                        stacked: true,
                        title: {{ display: true, text: 'Time (ms)' }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});

        new Chart(document.getElementById('timingPercentageChart'), {{
            type: 'bar',
            data: {{
                labels: {timing_scenarios_json},
                datasets: [
                    {{
                        label: 'Preprocessing',
                        data: {timing_preprocessing_pct_json},
                        backgroundColor: 'rgba(255, 99, 132, 0.7)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Input Preparation',
                        data: {timing_input_prep_pct_json},
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Model Forward',
                        data: {timing_model_forward_pct_json},
                        backgroundColor: 'rgba(255, 206, 86, 0.7)',
                        borderColor: 'rgba(255, 206, 86, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Postprocessing',
                        data: {timing_postprocessing_pct_json},
                        backgroundColor: 'rgba(75, 192, 192, 0.7)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Export',
                        data: {timing_export_pct_json},
                        backgroundColor: 'rgba(153, 102, 255, 0.7)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Time Breakdown by Stage (Percentage)',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    x: {{
                        stacked: true,
                        title: {{ display: true, text: 'Scenario' }}
                    }},
                    y: {{
                        stacked: true,
                        title: {{ display: true, text: 'Percentage (%)' }},
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
            """

        return f"""
        {timing_scripts}

        // Comparison Charts
        new Chart(document.getElementById('comparisonFpsChart'), {{
            type: 'bar',
            data: {{
                labels: {scenarios_json},
                datasets: [{{
                    label: 'FPS',
                    data: {fps_json},
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'FPS Comparison',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Frames Per Second' }}
                    }}
                }}
            }}
        }});

        new Chart(document.getElementById('comparisonTemporalChart'), {{
            type: 'bar',
            data: {{
                labels: {scenarios_json},
                datasets: [
                    {{
                        label: 'Smoothness (higher = better)',
                        data: {temporal_smoothness_json},
                        backgroundColor: 'rgba(40, 167, 69, 0.6)',
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 2,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Jitter (lower = better)',
                        data: {temporal_jitter_json},
                        backgroundColor: 'rgba(255, 193, 7, 0.6)',
                        borderColor: 'rgba(255, 193, 7, 1)',
                        borderWidth: 2,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Temporal Consistency (Dual Axis)',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        beginAtZero: true,
                        title: {{ display: true, text: 'Smoothness' }},
                        grid: {{ drawOnChartArea: true }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        beginAtZero: true,
                        title: {{ display: true, text: 'Jitter' }},
                        grid: {{ drawOnChartArea: false }}
                    }}
                }}
            }}
        }});

        new Chart(document.getElementById('comparisonStabilityChart'), {{
            type: 'bar',
            data: {{
                labels: {scenarios_json},
                datasets: [{{
                    label: 'Flicker Score (lower = better)',
                    data: {flicker_score_json},
                    backgroundColor: 'rgba(220, 53, 69, 0.6)',
                    borderColor: 'rgba(220, 53, 69, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Stability (Flicker)',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{ display: true, text: 'Flicker Score' }}
                    }}
                }}
            }}
        }});

        // Color palette for scenarios
        const colors = [
            'rgba(102, 126, 234, 0.8)',
            'rgba(220, 53, 69, 0.8)',
            'rgba(40, 167, 69, 0.8)',
            'rgba(255, 193, 7, 0.8)',
            'rgba(111, 66, 193, 0.8)',
            'rgba(23, 162, 184, 0.8)',
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 159, 64, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(201, 203, 207, 0.8)',
        ];

        new Chart(document.getElementById('comparisonLatencyMemoryChart'), {{
            type: 'scatter',
            data: {{
                datasets: {scenarios_json}.map((name, i) => ({{
                    label: name,
                    data: [{{
                        x: {latency_avg_json}[i],
                        y: {memory_avg_json}[i]
                    }}],
                    backgroundColor: colors[i % colors.length],
                    borderColor: colors[i % colors.length].replace('0.8', '1.0'),
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 10
                }}))
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Latency vs Memory',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                label += 'Latency: ' + context.parsed.x.toFixed(2) + 'ms, ';
                                label += 'Memory: ' + context.parsed.y.toFixed(2) + 'MB';
                                return label;
                            }}
                        }}
                    }},
                    legend: {{
                        display: true,
                        position: 'right',
                        labels: {{
                            boxWidth: 12,
                            padding: 8,
                            font: {{ size: 10 }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Average Latency (ms)' }},
                        beginAtZero: true
                    }},
                    y: {{
                        title: {{ display: true, text: 'Average Memory (MB)' }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});

        // FPS Chart
        new Chart(document.getElementById('fpsChart'), {{
            type: 'bar',
            data: {{
                labels: {scenarios_json},
                datasets: [{{
                    label: 'Average FPS',
                    data: {fps_json},
                    backgroundColor: 'rgba(102, 126, 234, 0.5)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Average FPS Comparison'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'FPS'
                        }}
                    }}
                }}
            }}
        }});

        // Latency Chart
        new Chart(document.getElementById('latencyChart'), {{
            type: 'bar',
            data: {{
                labels: {scenarios_json},
                datasets: [
                    {{
                        label: 'Avg Latency',
                        data: {latency_avg_json},
                        backgroundColor: 'rgba(40, 167, 69, 0.5)',
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 2
                    }},
                    {{
                        label: 'P95 Latency',
                        data: {latency_p95_json},
                        backgroundColor: 'rgba(255, 193, 7, 0.5)',
                        borderColor: 'rgba(255, 193, 7, 1)',
                        borderWidth: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Latency Comparison (ms)'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Milliseconds'
                        }}
                    }}
                }}
            }}
        }});

        // Memory Chart
        new Chart(document.getElementById('memoryChart'), {{
            type: 'bar',
            data: {{
                labels: {scenarios_json},
                datasets: [
                    {{
                        label: 'Avg Memory',
                        data: {memory_avg_json},
                        backgroundColor: 'rgba(220, 53, 69, 0.5)',
                        borderColor: 'rgba(220, 53, 69, 1)',
                        borderWidth: 2
                    }},
                    {{
                        label: 'Peak Memory',
                        data: {memory_max_json},
                        backgroundColor: 'rgba(111, 66, 193, 0.5)',
                        borderColor: 'rgba(111, 66, 193, 1)',
                        borderWidth: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Memory Usage (MB)'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Megabytes'
                        }}
                    }}
                }}
            }}
        }});

        // Frame Time Chart (all scenarios)
        new Chart(document.getElementById('frameTimeChart'), {{
            type: 'line',
            data: {{
                labels: {frame_indices_json},
                datasets: {frame_datasets_json}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Frame Processing Time (All Scenarios)'
                    }},
                    legend: {{
                        display: true,
                        position: 'top',
                        labels: {{
                            boxWidth: 12,
                            padding: 8,
                            font: {{ size: 10 }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Time (ms)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Frame Index'
                        }}
                    }}
                }}
            }}
        }});
        """

    def _generate_detailed_results(self, results: List[BenchmarkMetrics]) -> str:
        """Generate detailed results for each scenario."""
        sections = []

        for metric in results:
            stats_cards = f"""
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Avg FPS</div>
                    <div class="stat-value">{metric.avg_fps:.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Latency</div>
                    <div class="stat-value">{metric.avg_latency_ms:.1f}ms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Peak Memory</div>
                    <div class="stat-value">{metric.max_memory_mb:.1f}MB</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Temporal Jitter</div>
                    <div class="stat-value">{metric.temporal_jitter:.4f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Smoothness</div>
                    <div class="stat-value">{metric.temporal_smoothness:.3f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Flicker Score</div>
                    <div class="stat-value">{metric.flicker_score:.4f}</div>
                </div>
            </div>
            """

            # Check if temporal analysis images exist
            scenario_dir = self.output_dir / metric.scenario_name
            temporal_img = scenario_dir / "temporal_analysis.png"
            samples_img = scenario_dir / "temporal_samples.png"
            comparison_video = scenario_dir / "comparison.mp4"

            temporal_section = ""
            if temporal_img.exists():
                temporal_section = f"""
                <h4>Temporal Consistency Analysis</h4>
                <img src="{metric.scenario_name}/temporal_analysis.png" style="max-width: 100%; margin: 1rem 0;">
                """
            if samples_img.exists():
                temporal_section += f"""
                <h4>Sample Frames</h4>
                <img src="{metric.scenario_name}/temporal_samples.png" style="max-width: 100%; margin: 1rem 0;">
                """
            if comparison_video.exists():
                temporal_section += f"""
                <h4>Comparison Video</h4>
                <video controls style="max-width: 100%; margin: 1rem 0;">
                    <source src="{metric.scenario_name}/comparison.mp4" type="video/mp4">
                </video>
                """

            sections.append(f"""
            <div class="scenario-details">
                <h3>{metric.scenario_name}</h3>
                <p><strong>Device:</strong> {metric.device} | <strong>Frames:</strong> {metric.total_frames} | <strong>Time:</strong> {metric.total_time_s:.2f}s</p>
                {stats_cards}
                {temporal_section}
            </div>
            """)

        return f"""
        <div class="section">
            <h2>🔍 Detailed Results</h2>
            {''.join(sections)}
        </div>
        """
