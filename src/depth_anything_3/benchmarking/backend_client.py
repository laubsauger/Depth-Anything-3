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
Backend HTTP Client for Benchmarking

Provides HTTP client wrapper for backend inference API, enabling model reuse
across benchmark scenarios without reloading.
"""

import io
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import numpy as np
import requests
from PIL import Image

from depth_anything_3.api import Prediction


class BackendClient:
    """HTTP client for backend inference API."""

    def __init__(self, backend_url: str = "http://localhost:8008", timeout: int = 300):
        """
        Initialize backend client.

        Args:
            backend_url: Backend service URL
            timeout: Request timeout in seconds
        """
        self.backend_url = backend_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> bool:
        """
        Check if backend is healthy.

        Returns:
            True if backend is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.backend_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get backend status.

        Returns:
            Status dict with model info and device
        """
        response = self.session.get(
            f"{self.backend_url}/status",
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def inference(
        self,
        images: Union[List[Path], List[np.ndarray], List[Image.Image]],
        process_res: int = 504,
        infer_gs: bool = False,
        export_format: Optional[str] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = False,
    ) -> Prediction:
        """
        Run inference via backend API.

        Args:
            images: List of images (paths, numpy arrays, or PIL Images)
            process_res: Processing resolution
            infer_gs: Whether to infer Gaussian Splatting
            export_format: Export format (unused for backend)
            extrinsics: Camera extrinsics (N, 4, 4) - optional
            intrinsics: Camera intrinsics (N, 3, 3) - optional
            align_to_input_ext_scale: Align to input pose scale

        Returns:
            Prediction object with depth, confidence, poses, etc.
        """
        # Convert images to list of PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, Path):
                pil_images.append(Image.open(img))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        # Prepare multipart form data
        files = []
        for i, pil_img in enumerate(pil_images):
            # Convert to JPEG bytes
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='JPEG', quality=95)
            img_bytes.seek(0)
            files.append(('images', (f'image_{i}.jpg', img_bytes, 'image/jpeg')))

        # Prepare form data
        data = {
            'process_res': process_res,
            'infer_gs': infer_gs,
            'align_to_input_ext_scale': align_to_input_ext_scale,
        }

        # Add poses if provided
        if extrinsics is not None:
            # Convert to list for JSON serialization
            data['extrinsics'] = extrinsics.tolist()
        if intrinsics is not None:
            data['intrinsics'] = intrinsics.tolist()

        # Make request
        start_time = time.time()
        response = self.session.post(
            f"{self.backend_url}/inference",
            files=files,
            data=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        latency = time.time() - start_time

        # Parse response
        result = response.json()

        # Convert arrays from lists back to numpy
        depth = np.array(result['depth'])
        conf = np.array(result['confidence']) if result.get('confidence') else None
        extrinsics_out = np.array(result['extrinsics']) if result.get('extrinsics') else None
        intrinsics_out = np.array(result['intrinsics']) if result.get('intrinsics') else None
        sky_seg = np.array(result['sky_segmentation']) if result.get('sky_segmentation') else None

        # Create prediction object
        prediction = Prediction(
            depth=depth,
            conf=conf,
            extrinsics=extrinsics_out,
            intrinsics=intrinsics_out,
            sky_segmentation=sky_seg,
            gs_ply=None,  # Backend doesn't return raw PLY
            feat_vis=None,  # Backend doesn't return features
        )

        return prediction

    def close(self):
        """Close session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
