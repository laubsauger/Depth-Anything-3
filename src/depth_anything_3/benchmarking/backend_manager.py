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
Backend Service Manager

Manages backend service lifecycle for benchmarking, including automatic
startup and shutdown.
"""

import subprocess
import time
import signal
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import requests


class BackendManager:
    """Manages backend service lifecycle."""

    def __init__(
        self,
        model_dir: str,
        backend_url: str = "http://localhost:8008",
        device: str = "mps",
    ):
        """
        Initialize backend manager.

        Args:
            model_dir: Model directory for backend
            backend_url: Backend service URL
            device: Device to use (cuda, mps, cpu)
        """
        self.model_dir = model_dir
        self.backend_url = backend_url
        self.device = device
        self.process: Optional[subprocess.Popen] = None

        # Parse URL to get host and port
        parsed = urlparse(backend_url)
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 8008

    def is_running(self) -> bool:
        """Check if backend is running and responsive."""
        try:
            response = requests.get(
                f"{self.backend_url}/health",
                timeout=2
            )
            return response.status_code == 200
        except Exception:
            return False

    def start(self, timeout: int = 60) -> bool:
        """
        Start backend service.

        Args:
            timeout: Maximum time to wait for startup (seconds)

        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running():
            print(f"✅ Backend already running at {self.backend_url}")
            return True

        print(f"🚀 Starting backend service...")
        print(f"   Model: {self.model_dir}")
        print(f"   URL: {self.backend_url}")
        print(f"   Device: {self.device}")

        # Build command
        cmd = [
            sys.executable,  # Python interpreter
            "-m", "depth_anything_3.cli",
            "backend",
            "--model-dir", self.model_dir,
            "--host", self.host,
            "--port", str(self.port),
            "--device", self.device,
        ]

        # Start process
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False

        # Wait for backend to be ready
        print("⏳ Waiting for backend to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running():
                print(f"✅ Backend ready!")
                return True

            # Check if process died
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                print(f"❌ Backend process died:")
                print(f"   stdout: {stdout}")
                print(f"   stderr: {stderr}")
                return False

            time.sleep(1)

        print(f"❌ Backend failed to start within {timeout}s")
        self.stop()
        return False

    def stop(self):
        """Stop backend service."""
        if self.process is None:
            return

        print("🛑 Stopping backend service...")

        try:
            # Send SIGTERM
            self.process.send_signal(signal.SIGTERM)

            # Wait up to 10s for graceful shutdown
            try:
                self.process.wait(timeout=10)
                print("✅ Backend stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill
                print("⚠️  Backend didn't stop gracefully, force killing...")
                self.process.kill()
                self.process.wait()
                print("✅ Backend force killed")

        except Exception as e:
            print(f"⚠️  Error stopping backend: {e}")

        finally:
            self.process = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
