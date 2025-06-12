import argparse
import datetime
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import psutil
import requests
from tqdm import tqdm

# Attempt to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None # Placeholder

logger = logging.getLogger(__name__)
# Ensure a handler is configured for the logger if not already done by the main CLI
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define a type for a single benchmark result
BenchmarkResult = Dict[str, Any]
VideoMetadata = Dict[str, Any] # To store info about test videos

DEFAULT_ITERATIONS = 3
DEFAULT_WARMUP_ITERATIONS = 1

# Placeholder for BaseCommand if it's defined elsewhere in the CLI structure
# For standalone use or if BaseCommand is not yet defined in the project context.
class BaseCommand:
    """
    A base class for CLI commands, providing a common structure.
    Actual implementation would be part of the CLI framework.
    """
    @staticmethod
    def register_command(subparsers: argparse._SubParsersAction) -> None:
        """Registers the command and its arguments with the subparsers."""
        raise NotImplementedError("Subclasses must implement register_command.")

    def run_command(self, args: argparse.Namespace) -> None:
        """Executes the command's logic."""
        raise NotImplementedError("Subclasses must implement run_command.")


class BenchmarkCommand(BaseCommand):
    """
    CLI Command to benchmark feature extractors with different video formats and sizes.
    Measures processing time, memory usage, and GPU utilization.
    Generates a report of the benchmark results.
    """

    @staticmethod
    def register_command(subparsers: argparse._SubParsersAction) -> None:
        """Registers the benchmark command and its arguments."""
        parser = subparsers.add_parser(
            "benchmark",
            help="Benchmark feature extractor services.",
            description="Runs performance benchmarks against specified feature extractor services "
                        "using a collection of test videos or dummy text.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--extractors",
            nargs="+",
            required=True,
            help="List of feature extractor names to benchmark (e.g., clip2video clip dinov2). "
                 "These names must correspond to keys in the --service-config file.",
        )
        parser.add_argument(
            "--service-config",
            type=Path,
            required=True,
            help="Path to a JSON configuration file mapping extractor names to their service URLs "
                 "and input types. Example: "
                 "{'clip2video': {'url': 'http://localhost:8081', 'type': 'video', 'endpoint': '/get-video-feature'}, "
                 "'clip_text': {'url': 'http://localhost:8082', 'type': 'text', 'endpoint': '/get-text-feature'}}",
        )
        parser.add_argument(
            "--video-inputs",
            type=Path,
            help="Path to a directory containing test videos, or a JSON file listing video paths and their metadata. "
                 "Required if any 'video' type extractor is benchmarked. "
                 "If a directory, all common video files within will be used. "
                 "If a JSON file, it should be a list of objects, each with 'path' and optionally 'format', 'size_mb'.",
        )
        parser.add_argument(
            "--output-report-prefix",
            type=Path,
            default=Path("benchmark_report"),
            help="Prefix for the output report files (e.g., 'my_benchmark'). "
                 "'.json' and '.md' suffixes will be added.",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=DEFAULT_ITERATIONS,
            help="Number of times to run each test for averaging.",
        )
        parser.add_argument(
            "--warmup-iterations",
            type=int,
            default=DEFAULT_WARMUP_ITERATIONS,
            help="Number of warmup runs before actual measurement.",
        )
        parser.add_argument(
            "--dummy-text",
            type=str,
            default="A sample query text for benchmarking text feature extractors.",
            help="Dummy text to use for benchmarking text-based feature extractors."
        )
        parser.set_defaults(func=BenchmarkCommand.run_command_entry)

    @staticmethod
    def run_command_entry(args: argparse.Namespace) -> None:
        """Static entry point for the command, creating an instance and running it."""
        command = BenchmarkCommand(args)
        command.run_benchmark()

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.service_config: Dict[str, Dict[str, str]] = {}
        self.test_videos: List[VideoMetadata] = []
        self.results: List[BenchmarkResult] = []
        self.system_info: Dict[str, Any] = self._get_system_info()

        if PYNVML_AVAILABLE and pynvml:
            try:
                pynvml.nvmlInit()
                # Store GPU info directly in system_info
                gpu_device_info = self._get_gpu_device_info_pynvml()
                if gpu_device_info:
                    self.system_info["gpu_devices"] = gpu_device_info
                else:
                    self.system_info["gpu_devices"] = "No NVIDIA GPU detected or NVML error."
            except pynvml.NVMLError as e:
                logger.warning(f"Could not initialize NVML for GPU monitoring: {e}. GPU metrics will be unavailable.")
                self.system_info["gpu_devices"] = f"NVML Initialization Error: {e}"
        else:
            self.system_info["gpu_devices"] = "pynvml not available"


    def _load_service_config(self) -> bool:
        logger.info(f"Loading service configuration from: {self.args.service_config}")
        try:
            with open(self.args.service_config, "r", encoding="utf-8") as f:
                self.service_config = json.load(f)
            # Validate config structure
            for extractor_name, config in self.service_config.items():
                if not all(k in config for k in ["url", "type", "endpoint"]):
                    logger.error(f"Invalid configuration for extractor '{extractor_name}'. "
                                 "Must include 'url', 'type', and 'endpoint'.")
                    return False
                if config["type"] not in ["video", "text"]:
                    logger.error(f"Invalid type '{config['type']}' for extractor '{extractor_name}'. "
                                 "Must be 'video' or 'text'.")
                    return False
            logger.info("Service configuration loaded successfully.")
            return True
        except FileNotFoundError:
            logger.error(f"Service configuration file not found: {self.args.service_config}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding service configuration JSON from {self.args.service_config}: {e}")
            return False

    def _collect_test_videos(self) -> bool:
        if not self.args.video_inputs:
            # This is not an error if no video extractors are selected.
            # This will be checked later.
            logger.info("No video input path provided. Video tests will be skipped if video extractors are selected.")
            return True # Allow proceeding if only text extractors are benchmarked

        logger.info(f"Collecting test videos from: {self.args.video_inputs}")
        if not self.args.video_inputs.exists():
            logger.error(f"Video input path not found: {self.args.video_inputs}")
            return False

        if self.args.video_inputs.is_dir():
            video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".ts", ".mts", ".m2ts", ".mpg", ".mpeg", ".m4v"]
            for video_file in self.args.video_inputs.rglob("*"): # Recursive glob
                if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                    try:
                        size_mb = video_file.stat().st_size / (1024 * 1024)
                        self.test_videos.append({
                            "path": str(video_file.resolve()),
                            "name": video_file.name,
                            "format": video_file.suffix.lower().lstrip('.'),
                            "size_mb": round(size_mb, 3),
                        })
                    except Exception as e:
                        logger.warning(f"Could not get metadata for video {video_file}: {e}")
        elif self.args.video_inputs.is_file() and self.args.video_inputs.suffix.lower() == ".json":
            try:
                with open(self.args.video_inputs, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        logger.error("Video input JSON file must contain a list of video metadata objects.")
                        return False
                    for item_idx, item in enumerate(data):
                        if "path" not in item or not isinstance(item["path"], str):
                            logger.warning(f"Skipping item #{item_idx} in JSON due to missing or invalid 'path': {item}")
                            continue
                        video_path = Path(item["path"])
                        if not video_path.exists() or not video_path.is_file():
                            logger.warning(f"Video file specified in JSON not found or not a file: {video_path}. Skipping.")
                            continue
                        try:
                            size_mb = item.get("size_mb")
                            if size_mb is None:
                                size_mb = round(video_path.stat().st_size / (1024*1024), 3)

                            self.test_videos.append({
                                "path": str(video_path.resolve()),
                                "name": item.get("name", video_path.name),
                                "format": item.get("format", video_path.suffix.lower().lstrip('.')),
                                "size_mb": size_mb,
                            })
                        except Exception as e:
                             logger.warning(f"Could not process video entry {item.get('path', 'Unknown')}: {e}")

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding video inputs JSON from {self.args.video_inputs}: {e}")
                return False
        else:
            logger.error(f"Invalid video_inputs path: {self.args.video_inputs}. Must be a directory or a .json file.")
            return False

        if not self.test_videos:
            logger.warning("No test videos found or collected from the provided path.")
            # This is not necessarily a fatal error if only text extractors are chosen.
        else:
            logger.info(f"Collected {len(self.test_videos)} test videos.")
        return True

    def _get_system_info(self) -> Dict[str, Any]:
        logger.info("Gathering system information...")
        info: Dict[str, Any] = {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_name": "N/A", # Requires platform-specific way or library like cpuinfo
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }
        try:
            # Attempt to get CPU name using lscpu (Linux specific)
            if platform.system() == "Linux":
                result = subprocess.run(['lscpu'], capture_output=True, text=True, check=False)
                for line in result.stdout.splitlines():
                    if "Model name:" in line:
                        info["cpu_name"] = line.split("Model name:")[1].strip()
                        break
        except Exception:
            pass # Ignore if lscpu fails or not available
        return info

    def _get_gpu_device_info_pynvml(self) -> List[Dict[str, Any]]:
        if not PYNVML_AVAILABLE or pynvml is None:
            return []
        
        gpu_info_list: List[Dict[str, Any]] = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name_bytes = pynvml.nvmlDeviceGetName(handle)
                name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else str(name_bytes)
                total_memory_gb = round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3), 2)
                gpu_info_list.append({
                    "id": i,
                    "name": name,
                    "total_memory_gb": total_memory_gb,
                })
            return gpu_info_list
        except pynvml.NVMLError as e:
            logger.warning(f"Could not retrieve GPU device info via NVML: {e}")
            return [{"error": str(e)}] # Return error state if query fails
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error getting GPU device info: {e}")
            return [{"error": f"Unexpected error: {str(e)}"}]


    def _measure_gpu_util_pynvml(self) -> Optional[Dict[str, Any]]:
        if not PYNVML_AVAILABLE or pynvml is None or not self.system_info.get("gpu_devices") \
           or isinstance(self.system_info.get("gpu_devices"), str) \
           or (isinstance(self.system_info.get("gpu_devices"), list) and not self.system_info.get("gpu_devices")) \
           or (isinstance(self.system_info.get("gpu_devices"), list) and "error" in self.system_info.get("gpu_devices")[0]): # type: ignore
            return None
        
        # Reporting for the first GPU for simplicity.
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "gpu_utilization_percent": util.gpu,
                "gpu_memory_used_percent": round((mem_info.used / mem_info.total) * 100, 2) if mem_info.total > 0 else 0,
                "gpu_memory_used_gb": round(mem_info.used / (1024**3), 2),
            }
        except pynvml.NVMLError as e:
            logger.debug(f"Failed to get GPU utilization snapshot: {e}") # Debug as this can be frequent
            return {"error": str(e)}
        except Exception as e:
            logger.warning(f"Unexpected error measuring GPU utilization: {e}")
            return {"error": f"Unexpected error: {str(e)}"}


    def _ping_service(self, service_url: str) -> bool:
        ping_url = f"{service_url.rstrip('/')}/ping"
        try:
            response = requests.get(ping_url, timeout=5) # 5 second timeout for ping
            response.raise_for_status() # Check for HTTP errors
            logger.info(f"Service at {service_url} is responsive (ping successful).")
            return True
        except requests.exceptions.Timeout:
            logger.error(f"Ping timeout for service at {service_url}.")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"Ping connection error for service at {service_url}.")
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"Ping HTTP error for service at {service_url}: {e.response.status_code}")
            return False
        except requests.RequestException as e:
            logger.error(f"Ping request exception for service at {service_url}: {e}")
            return False

    def _run_single_test(
        self, extractor_name: str, service_config: Dict[str, str], video_meta: Optional[VideoMetadata]
    ) -> BenchmarkResult:
        """Runs a single test iteration for one video/text against one extractor."""
        service_url = service_config["url"]
        service_type = service_config["type"]
        service_endpoint = service_config["endpoint"]
        
        start_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        initial_mem_rss_mb = process.memory_info().rss / (1024 * 1024)
        
        gpu_metrics_start = self._measure_gpu_util_pynvml()
        
        error_message: Optional[str] = None
        status_code: Optional[int] = None
        response_content_summary: str = "N/A"

        full_endpoint_url = f"{service_url.rstrip('/')}{service_endpoint}"

        try:
            if service_type == "video":
                if not video_meta or "path" not in video_meta or not Path(video_meta["path"]).exists():
                    raise FileNotFoundError(f"Video file not found or path missing: {video_meta.get('path') if video_meta else 'N/A'}")
                
                video_file_path = Path(video_meta["path"])
                with open(video_file_path, "rb") as f:
                    # 'file' is a common key for file uploads
                    files = {"file": (video_meta.get("name", video_file_path.name), f)}
                    response = requests.post(full_endpoint_url, files=files, timeout=300) # 5 min timeout for video processing
            elif service_type == "text":
                payload = {"text": self.args.dummy_text}
                response = requests.post(full_endpoint_url, json=payload, timeout=60) # 1 min timeout for text
            else:
                raise ValueError(f"Internal error: Unknown service type '{service_type}'")

            status_code = response.status_code
            response.raise_for_status()
            try:
                response_json = response.json()
                if response_json and isinstance(response_json.get("embedding"), list):
                    response_content_summary = f"Embedding found (len: {len(response_json['embedding'])})"
                elif response_json:
                    response_content_summary = f"JSON response (no embedding key or not list): {str(response_json)[:100]}"
                else:
                    response_content_summary = "Empty JSON response"
            except json.JSONDecodeError:
                response_content_summary = f"Non-JSON response (first 100 chars): {response.text[:100]}"


        except requests.exceptions.Timeout:
            error_message = "Request timed out"
            logger.warning(f"Timeout for {extractor_name} with {video_meta['name'] if video_meta else 'dummy text'}.")
        except requests.exceptions.RequestException as e:
            error_message = f"RequestException: {type(e).__name__} - {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                try:
                    error_detail = e.response.json().get("error", e.response.text[:100])
                    error_message = f"HTTP {status_code}: {error_detail}"
                except json.JSONDecodeError:
                     error_message = f"HTTP {status_code}: {e.response.text[:100]}"
            logger.warning(f"RequestException for {extractor_name} with {video_meta['name'] if video_meta else 'dummy text'}: {error_message}")
        except FileNotFoundError as e:
            error_message = f"FileNotFoundError: {str(e)}"
            logger.error(f"File not found for benchmark: {e}")
        except Exception as e:
            error_message = f"Unexpected {type(e).__name__}: {str(e)}"
            logger.error(f"Unexpected error during test for {extractor_name}: {e}", exc_info=True)

        end_time = time.perf_counter()
        final_mem_rss_mb = process.memory_info().rss / (1024 * 1024)
        gpu_metrics_end = self._measure_gpu_util_pynvml()

        result: BenchmarkResult = {
            "extractor": extractor_name,
            "service_type": service_type,
            "input_name": video_meta["name"] if video_meta and service_type == "video" else (self.args.dummy_text[:30]+"..." if service_type == "text" else "N/A"),
            "input_path": video_meta["path"] if video_meta and service_type == "video" else "N/A",
            "input_format": video_meta.get("format", "N/A") if video_meta and service_type == "video" else "N/A",
            "input_size_mb": video_meta.get("size_mb", 0) if video_meta and service_type == "video" else 0,
            "processing_time_seconds": round(end_time - start_time, 4),
            "client_mem_rss_initial_mb": round(initial_mem_rss_mb, 3),
            "client_mem_rss_final_mb": round(final_mem_rss_mb, 3),
            "client_mem_rss_delta_mb": round(final_mem_rss_mb - initial_mem_rss_mb, 3),
            "status_code": status_code,
            "error": error_message,
            "response_summary": response_content_summary,
        }
        if gpu_metrics_start and not gpu_metrics_start.get("error"):
            result["gpu_util_start_percent"] = gpu_metrics_start.get("gpu_utilization_percent")
            result["gpu_mem_used_start_gb"] = gpu_metrics_start.get("gpu_memory_used_gb")
        if gpu_metrics_end and not gpu_metrics_end.get("error"):
            result["gpu_util_end_percent"] = gpu_metrics_end.get("gpu_utilization_percent")
            result["gpu_mem_used_end_gb"] = gpu_metrics_end.get("gpu_memory_used_gb")
        
        return result

    def run_benchmark(self) -> None:
        logger.info("Starting Visione Benchmark...")
        if not self._load_service_config():
            logger.error("Failed to load service configuration. Aborting benchmark.")
            return
        if not self._collect_test_videos(): # This now handles the case where video_inputs is None
            # If _collect_test_videos returns False, it means video_inputs was provided but invalid.
            logger.error("Failed to collect test videos. Aborting benchmark.")
            return

        # Check if video inputs are required but not available
        needs_video_inputs = any(
            self.service_config.get(name, {}).get("type") == "video" for name in self.args.extractors
        )
        if needs_video_inputs and not self.test_videos:
            logger.error("Video-type extractors are selected, but no test videos were collected. "
                         "Please provide a valid --video-inputs path. Aborting.")
            return


        total_tests_to_run = 0
        for extractor_name in self.args.extractors:
            config = self.service_config.get(extractor_name)
            if not config or extractor_name not in self.service_config: # Check if extractor is in loaded config
                logger.warning(f"Configuration for extractor '{extractor_name}' not found in service config. Skipping.")
                continue
            
            num_inputs_for_extractor = 0
            if config["type"] == "video":
                num_inputs_for_extractor = len(self.test_videos)
            elif config["type"] == "text":
                num_inputs_for_extractor = 1 # One dummy text input
            
            if num_inputs_for_extractor > 0:
                 total_tests_to_run += num_inputs_for_extractor * (self.args.iterations + self.args.warmup_iterations)
        
        if total_tests_to_run == 0:
            logger.info("No tests to run based on selected extractors and available inputs. Exiting.")
            self._save_report() # Save an empty/minimal report
            return

        progress_bar = tqdm(total=total_tests_to_run, desc="Running benchmarks", unit="test", dynamic_ncols=True)

        for extractor_name in self.args.extractors:
            config = self.service_config.get(extractor_name)
            if not config or extractor_name not in self.service_config:
                continue # Already logged

            service_url = config["url"]
            service_type = config["type"]

            logger.info(f"\n--- Benchmarking Extractor: {extractor_name} ({service_url}, type: {service_type}) ---")
            if not self._ping_service(service_url):
                logger.error(f"Skipping benchmark for {extractor_name} as it's not responsive.")
                num_skipped_for_this_extractor = 0
                if service_type == "video":
                    num_skipped_for_this_extractor = len(self.test_videos) * (self.args.iterations + self.args.warmup_iterations)
                elif service_type == "text":
                    num_skipped_for_this_extractor = (self.args.iterations + self.args.warmup_iterations)
                progress_bar.update(num_skipped_for_this_extractor)
                self.results.append({
                    "extractor": extractor_name, "error": "Service not responsive at ping", "skipped_all_tests": True
                })
                continue

            inputs_to_process: List[Optional[VideoMetadata]]
            if service_type == "video":
                if not self.test_videos:
                    logger.info(f"No videos to test for video extractor '{extractor_name}'. Skipping its tests.")
                    continue
                inputs_to_process = self.test_videos
            elif service_type == "text":
                inputs_to_process = [None] # Represents the single dummy text input
            else: # Should be caught by config validation
                logger.error(f"Internal error: Unknown service type '{service_type}' for {extractor_name}")
                continue

            for current_input_meta in inputs_to_process:
                input_description = current_input_meta["name"] if current_input_meta else "dummy text"
                progress_bar.set_description(f"Benchmarking {extractor_name} on {input_description[:20]}")

                # Warmup iterations
                if self.args.warmup_iterations > 0:
                    logger.debug(f"Running {self.args.warmup_iterations} warmup iterations for {input_description}...")
                    for _ in range(self.args.warmup_iterations):
                        self._run_single_test(extractor_name, config, current_input_meta)
                        progress_bar.update(1)
                
                iteration_results_for_input: List[BenchmarkResult] = []
                logger.debug(f"Running {self.args.iterations} benchmark iterations for {input_description}...")
                for i in range(self.args.iterations):
                    logger.debug(f"Iteration {i+1}/{self.args.iterations} for {input_description}")
                    res = self._run_single_test(extractor_name, config, current_input_meta)
                    iteration_results_for_input.append(res)
                    progress_bar.update(1)
                
                if iteration_results_for_input:
                    agg_res = self._aggregate_iteration_results(iteration_results_for_input)
                    self.results.append(agg_res)
        
        progress_bar.close()
        logger.info("\nBenchmark run finished.")
        self._save_report()

        if PYNVML_AVAILABLE and pynvml:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                logger.warning(f"Error during NVML shutdown: {e}")
            except Exception as e: # Catch other potential errors if nvmlShutdown is problematic
                logger.warning(f"Unexpected error during NVML shutdown: {e}")


    def _aggregate_iteration_results(self, iter_results: List[BenchmarkResult]) -> BenchmarkResult:
        """Aggregates results from multiple iterations for a single test case (extractor + input)."""
        if not iter_results:
            # This case should ideally not be reached if called correctly.
            return {"error": "No iteration results to aggregate."}

        first_res = iter_results[0]
        # Core identifying information from the first result
        agg: BenchmarkResult = {
            "extractor": first_res["extractor"],
            "service_type": first_res["service_type"],
            "input_name": first_res["input_name"],
            "input_path": first_res["input_path"],
            "input_format": first_res["input_format"],
            "input_size_mb": first_res["input_size_mb"],
            "iterations_run": len(iter_results),
            "successful_iterations": sum(1 for r in iter_results if not r["error"]),
            "failed_iterations": sum(1 for r in iter_results if r["error"]),
            "error_messages_summary": list(set(r["error"] for r in iter_results if r["error"])),
        }

        # Filter to successful iterations for performance metrics
        successful_iter_results = [r for r in iter_results if not r["error"]]

        if successful_iter_results:
            # Average numerical fields from successful runs
            fields_to_average: List[str] = [
                "processing_time_seconds", "client_mem_rss_delta_mb",
                "gpu_util_start_percent", "gpu_util_end_percent",
                "gpu_mem_used_start_gb", "gpu_mem_used_end_gb"
            ]
            for field_key in fields_to_average:
                values = [r.get(field_key) for r in successful_iter_results if r.get(field_key) is not None and not isinstance(r.get(field_key), dict)] # Ensure not error dict
                if values:
                    agg[f"avg_{field_key}"] = round(sum(values) / len(values), 4)
                    agg[f"min_{field_key}"] = round(min(values), 4)
                    agg[f"max_{field_key}"] = round(max(values), 4)
                    # Standard deviation could also be useful here
                    # agg[f"std_{field_key}"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
                else:
                    agg[f"avg_{field_key}"] = None # Or 0, or "N/A" depending on preference

            avg_time = agg.get("avg_processing_time_seconds")
            if avg_time and avg_time > 0:
                if first_res["service_type"] == "video" and isinstance(first_res["input_size_mb"], (int, float)) and first_res["input_size_mb"] > 0:
                    agg["throughput_mb_per_second"] = round(first_res["input_size_mb"] / avg_time, 3)
                else: # Text or 0-size video
                    agg["throughput_items_per_second"] = round(1.0 / avg_time, 3)
        else: # All iterations failed
            agg["avg_processing_time_seconds"] = None # Or some indicator of failure
            logger.warning(f"All iterations failed for {agg['extractor']} on {agg['input_name']}. Performance metrics will be null.")
            
        return agg

    def _save_report(self) -> None:
        # Ensure output directory for report exists
        self.args.output_report_prefix.parent.mkdir(parents=True, exist_ok=True)

        report_data: Dict[str, Any] = {
            "benchmark_metadata": {
                "report_generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "benchmark_run_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"), # Added microsecs for more uniqueness
                "benchmark_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()},
                "system_info": self.system_info,
                "service_config_used": self.service_config,
            },
            "benchmark_results": self.results,
        }
        
        # Save JSON report
        json_report_path = self.args.output_report_prefix.with_suffix(".json")
        logger.info(f"Saving JSON benchmark report to: {json_report_path}")
        try:
            with open(json_report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4, default=str) # default=str for any other non-serializable
            logger.info("JSON report saved successfully.")
        except IOError as e:
            logger.error(f"Failed to write JSON report to {json_report_path}: {e}")
        except TypeError as e:
            logger.error(f"TypeError during JSON serialization of report: {e}")

        # Save Markdown report
        md_report_path = self.args.output_report_prefix.with_suffix(".md")
        logger.info(f"Saving Markdown benchmark report to: {md_report_path}")
        try:
            with open(md_report_path, "w", encoding="utf-8") as f:
                f.write(self._generate_markdown_report(report_data))
            logger.info("Markdown report saved successfully.")
        except IOError as e:
            logger.error(f"Failed to write Markdown report to {md_report_path}: {e}")

    def _format_value(self, value: Any, precision: int = 2, default_na: str = "N/A") -> str:
        """Helper to format values for Markdown, handling None."""
        if value is None:
            return default_na
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return str(value)

    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        meta = report_data["benchmark_metadata"]
        results = report_data["benchmark_results"]

        md_lines: List[str] = []
        md_lines.append(f"# Visione Performance Benchmark Report")
        md_lines.append(f"- **Run ID**: {meta['benchmark_run_id']}")
        md_lines.append(f"- **Generated At (UTC)**: {meta['report_generated_at_utc']}")
        
        md_lines.append("\n## System Information")
        for key, value in meta["system_info"].items():
            display_key = key.replace('_', ' ').title()
            if key == "gpu_devices":
                md_lines.append(f"- **{display_key}**:")
                if isinstance(value, list) and value:
                    if isinstance(value[0], dict) and "error" in value[0]:
                         md_lines.append(f"  - Error: {value[0]['error']}")
                    else:
                        for gpu in value:
                            md_lines.append(f"  - ID {gpu.get('id', 'N/A')}: {gpu.get('name', 'N/A')}, "
                                          f"Total Memory: {self._format_value(gpu.get('total_memory_gb'))} GB")
                elif isinstance(value, str): # Handle simple string messages like "pynvml not available"
                     md_lines.append(f"  - {value}")
                else:
                    md_lines.append(f"  - No GPU devices reported or N/A.")
            else:
                md_lines.append(f"- **{display_key}**: {value}")

        md_lines.append("\n## Benchmark Configuration")
        for key, value in meta["benchmark_args"].items():
             md_lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        
        md_lines.append("\n## Service Configuration Used")
        for name, conf in meta["service_config_used"].items():
            md_lines.append(f"- **Extractor `{name}`**:")
            md_lines.append(f"  - URL: `{conf['url']}`")
            md_lines.append(f"  - Type: `{conf['type']}`")
            md_lines.append(f"  - Endpoint: `{conf['endpoint']}`")


        md_lines.append("\n## Benchmark Results Summary")
        if not results:
            md_lines.append("No benchmark results recorded.")
        else:
            # Define table headers
            headers = [
                "Extractor", "Input Name", "Input Size (MB)", "Avg Time (s)", 
                "Throughput", "Success/Total Iter", "Client Mem Δ (MB)",
                "Avg GPU Util Start/End (%)", "Avg GPU Mem Start/End (GB)"
            ]
            md_lines.append(f"| {' | '.join(headers)} |")
            md_lines.append(f"|{'|:'.join(['---'] * len(headers))}-|") # Separator line

            for res in results:
                if res.get("skipped_all_tests"): # Handle cases where an entire extractor was skipped
                    row = [
                        res['extractor'], 
                        f"SKIPPED ({res.get('error', 'Unknown reason')})",
                        "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
                    ]
                    md_lines.append(f"| {' | '.join(row)} |")
                    continue

                throughput_str = "N/A"
                if res.get("throughput_mb_per_second") is not None:
                    throughput_str = f"{self._format_value(res['throughput_mb_per_second'], 2)} MB/s"
                elif res.get("throughput_items_per_second") is not None:
                    throughput_str = f"{self._format_value(res['throughput_items_per_second'], 2)} items/s"
                
                gpu_util_str = "N/A"
                if res.get("avg_gpu_util_start_percent") is not None and res.get("avg_gpu_util_end_percent") is not None:
                     gpu_util_str = (f"{self._format_value(res['avg_gpu_util_start_percent'], 1)} / "
                                     f"{self._format_value(res['avg_gpu_util_end_percent'], 1)}")

                gpu_mem_str = "N/A"
                if res.get("avg_gpu_mem_used_start_gb") is not None and res.get("avg_gpu_mem_used_end_gb") is not None:
                     gpu_mem_str = (f"{self._format_value(res['avg_gpu_mem_used_start_gb'], 2)} / "
                                    f"{self._format_value(res['avg_gpu_mem_used_end_gb'], 2)}")
                
                input_name_display = res.get('input_name', 'N/A')
                if len(input_name_display) > 30: # Truncate long input names for table
                    input_name_display = input_name_display[:27] + "..."

                row_data = [
                    res.get('extractor', 'N/A'),
                    input_name_display,
                    self._format_value(res.get('input_size_mb'), 2),
                    self._format_value(res.get('avg_processing_time_seconds'), 3),
                    throughput_str,
                    f"{res.get('successful_iterations', 0)}/{res.get('iterations_run', 0)}",
                    self._format_value(res.get('avg_client_mem_rss_delta_mb'), 2),
                    gpu_util_str,
                    gpu_mem_str
                ]
                md_lines.append(f"| {' | '.join(row_data)} |")
        
        md_lines.append("\n## Detailed Error Messages (if any)")
        errors_found_in_report = False
        for res_idx, res in enumerate(results):
            if res.get("failed_iterations", 0) > 0 and res.get("error_messages_summary"):
                if not errors_found_in_report: errors_found_in_report = True
                md_lines.append(f"\n### Errors for Test Case #{res_idx+1} ({res.get('extractor','N/A')} on {res.get('input_name','N/A')}):")
                for err_msg in res["error_messages_summary"]:
                    md_lines.append(f"- `{err_msg}`") # Code block for error messages
        if not errors_found_in_report:
            md_lines.append("No errors recorded across successful iterations, or all tests were skipped.")

        return "\n".join(md_lines)


# Main guard for direct script execution (e.g., for testing the command module)
if __name__ == "__main__":
    # This block allows for testing the benchmark command module directly.
    # In a real CLI application, `BenchmarkCommand.register_command` would be called
    # by the main CLI entry point, and `argparse` would handle dispatching.
    
    # Create a dummy parser and subparsers to simulate CLI structure
    parser = argparse.ArgumentParser(description="Visione Benchmark CLI (Test Runner)")
    subparsers = parser.add_subparsers(title="commands", dest="command_name", help="Available commands")
    
    # Register the benchmark command
    BenchmarkCommand.register_command(subparsers)
    
    # --- Example: Construct arguments for a test run ---
    # Create dummy files and directories for a self-contained example.
    # Note: For this example to *actually* run and produce meaningful results,
    #       you would need:
    #       1. Dummy/real feature extractor services running at the specified URLs.
    #       2. Actual video files in the `dummy_video_dir_for_benchmark`.
    
    current_script_dir = Path(__file__).parent
    dummy_root = current_script_dir / "benchmark_test_workspace"
    dummy_root.mkdir(exist_ok=True)

    dummy_service_config_path = dummy_root / "dummy_services.json"
    dummy_video_dir_path = dummy_root / "test_videos"
    dummy_video_dir_path.mkdir(exist_ok=True)
    dummy_report_prefix = dummy_root / "example_report"

    # Create a dummy service config file
    dummy_services_data = {
        "dummy_video_svc": {
            "url": "http://localhost:12345", # Replace with a real or mock server URL
            "type": "video",
            "endpoint": "/get-video-feature"
        },
        "dummy_text_svc": {
            "url": "http://localhost:12346", # Replace with a real or mock server URL
            "type": "text",
            "endpoint": "/get-text-feature"
        }
    }
    with open(dummy_service_config_path, "w", encoding="utf-8") as f_cfg:
        json.dump(dummy_services_data, f_cfg, indent=2)

    # Create a dummy video file (e.g., a very small, short MP4)
    # This step is crucial. Without a valid video, video tests will fail.
    # For a quick test, you might copy a small existing mp4 here.
    # Example: `dummy_video_dir_path / "small_test.mp4"`
    # If ffmpeg is available, you could generate one:
    # `ffmpeg -f lavfi -i testsrc=duration=1:size=qcif:rate=1 -c:v libx264 -preset ultrafast -tune zerolatency dummy_video_dir_path/small_test.mp4`
    # For this script, we'll assume the user places a video there or handles FileNotFoundError.
    # Let's create an empty file to make the path valid for collection, but tests will fail on it.
    (dummy_video_dir_path / "empty_test.mp4").touch()


    # Simulate command-line arguments
    # Note: The dummy service URLs will likely cause connection errors unless mock servers are running.
    test_args_str = (
        f"benchmark "
        f"--extractors dummy_video_svc dummy_text_svc "
        f"--service-config {str(dummy_service_config_path)} "
        f"--video-inputs {str(dummy_video_dir_path)} "
        f"--output-report-prefix {str(dummy_report_prefix)} "
        f"--iterations 1 "
        f"--warmup-iterations 0"
    )
    
    logger.info(f"Simulating CLI execution with args: {test_args_str}")
    
    try:
        parsed_args = parser.parse_args(test_args_str.split())
        if hasattr(parsed_args, "func"):
            parsed_args.func(parsed_args) # Calls BenchmarkCommand.run_command_entry
        else:
            # This case should not be reached if 'benchmark' is a registered subparser command
            parser.print_help()
            logger.error("No function associated with parsed arguments. Subparser setup might be incorrect.")
    except SystemExit as e:
        # Argparse calls sys.exit on error or --help. Catch it for cleaner test runs.
        if e.code != 0: # Re-raise if it's an actual error exit
            logger.error(f"Argparse exited with code {e.code}")
            # raise # Uncomment to fail test on argparse error
    except Exception as e:
        logger.error(f"An error occurred during test execution: {e}", exc_info=True)
    finally:
        # Clean up dummy files (optional, for keeping test environment clean)
        # logger.info("Cleaning up dummy benchmark files...")
        # if dummy_service_config_path.exists(): dummy_service_config_path.unlink()
        # if (dummy_video_dir_path / "empty_test.mp4").exists(): (dummy_video_dir_path / "empty_test.mp4").unlink()
        # if dummy_video_dir_path.exists(): dummy_video_dir_path.rmdir() # Only if empty
        # Reports (example_report.json, example_report.md) are left for inspection.
        logger.info(f"Dummy benchmark run finished. Check reports at: {dummy_report_prefix}.[json|md]")
