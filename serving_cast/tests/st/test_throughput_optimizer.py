# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
import subprocess
from unittest import TestCase

from serving_cast.service.optimizer_summary import SHOW_COLUMNS


class TestThroughputOptimizer(TestCase):
    """Performance analysis script system test class"""

    def _run_throughput_optimizer(self, args, check=True):
        """Run throughput_optimizer script using module execution"""
        cmd = ["python", "-m", "cli.inference.throughput_optimizer"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result

    def _validate_table_structure(
        self, output_text, required_columns, table_start_pattern
    ):
        """Validate the overall table structure and format"""
        # Check for required sections
        required_sections = [
            "Input Configuration:",
            "Overall Best Configuration:",
        ]

        for section in required_sections:
            self.assertIsNotNone(
                re.search(section, output_text),
                f"Required section '{section}' not found in output",
            )

        # Check for table header columns
        header_line = None

        for line in output_text.split("\n"):
            if all(col in line for col in required_columns):
                header_line = line
                break

        self.assertIsNotNone(
            header_line, "Table header with required columns not found"
        )

        # Check for table borders (prettytable format)
        border_pattern = r"\+-+\+"
        borders = re.findall(border_pattern, output_text)
        self.assertGreaterEqual(
            len(borders), 2, "Table borders not found or incomplete"
        )

        # Check for data rows in table format
        data_row_pattern = r"\|\s*\d+\s*\|.*\|"
        data_rows = re.findall(data_row_pattern, output_text)
        self.assertGreaterEqual(len(data_rows), 1, "Table data rows not found")

        # Check for the specific table format
        self.assertIsNotNone(
            re.search(table_start_pattern, output_text),
            "Configurations table title not found",
        )

        # Check for throughput values in table
        throughput_pattern = r"\|\s*\d+\s*\|\s*[\d.]+\s*"
        throughput_matches = re.findall(throughput_pattern, output_text)
        self.assertGreaterEqual(
            len(throughput_matches), 1, "Throughput values not found in table"
        )

    def test_aggregation_functionality_with_output_validation(self):
        """Test aggregation functionality with comprehensive output validation"""
        args = [
            "--input-length=3500",
            "--output-length=1500",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            "--tpot-limits=50",
            "--compile",
        ]

        # Execute command
        result = self._run_throughput_optimizer(args, check=False)

        # Basic execution check
        if result.returncode != 0:
            self.fail(
                f"Script execution failed with return code {result.returncode}: {result.stderr}"
            )

        # Combine stdout and stderr for analysis
        full_output = result.stdout + result.stderr

        # Validate table structure
        required_columns = SHOW_COLUMNS
        table_start_pattern = r"Top \d Aggregation Configurations:"
        self._validate_table_structure(
            full_output, required_columns, table_start_pattern
        )

    def test_disaggregation_prefill_only_with_output_validation(self):
        """Test disaggregation prefill only functionality with comprehensive output validation"""
        args = [
            "--input-length=1024",
            "--output-length=1024",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            "--ttft-limits=1000",
            "--compile",
            "--disagg",
        ]

        # Execute command
        result = self._run_throughput_optimizer(args, check=False)

        # Basic execution check
        if result.returncode != 0:
            self.fail(
                f"Script execution failed with return code {result.returncode}: {result.stderr}"
            )

        # Combine stdout and stderr for analysis
        full_output = result.stdout + result.stderr
        # Validate table structure
        local_columns = SHOW_COLUMNS.copy()
        local_columns.remove("TPOT (ms)")
        table_start_pattern = r"Top \d Disaggregation \(Prefill\) Configurations:"
        self._validate_table_structure(full_output, local_columns, table_start_pattern)

    def test_disaggregation_decode_only_with_output_validation(self):
        """Test disaggregation decode only functionality with comprehensive output validation"""
        args = [
            "--input-length=1024",
            "--output-length=1024",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            "--num-devices=8",
            "--tpot-limits=50",
            "--compile",
            "--disagg",
            "--tp-sizes",
            "2",
            "4",
            "--batch-range",
            "1",
            "8",
        ]

        # Execute command
        result = self._run_throughput_optimizer(args, check=False)

        # Basic execution check
        if result.returncode != 0:
            self.fail(
                f"Script execution failed with return code {result.returncode}: {result.stderr}"
            )

        # Combine stdout and stderr for analysis
        full_output = result.stdout + result.stderr
        # Validate table structure
        local_columns = SHOW_COLUMNS.copy()
        local_columns.remove("TTFT (ms)")
        table_start_pattern = r"Top \d Disaggregation \(Decode\) Configurations:"
        self._validate_table_structure(full_output, local_columns, table_start_pattern)
