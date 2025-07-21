"""
Base classes for feature extractors that process images and videos.

This module provides abstract base classes for implementing feature extractors
for both image and video data, with support for batch processing, GPU acceleration,
and asynchronous operation.
"""
import asyncio
import csv
import functools
import itertools
import logging
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, AsyncGenerator, AsyncIterable, Callable, Dict, Generic, Iterable, 
    Iterator, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, cast
)

import more_itertools
import numpy as np
import torch
from tqdm.asyncio import tqdm as async_tqdm
from tqdm.auto import tqdm

from .savers import AbstractSaver, GzipJsonlFile, HDF5File, SaverError

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
PathLike = Union[str, Path]
VideoID = str
ImageID = str
ImagePath = Path
RecordType = Dict[str, Any]
ImageInfo = Tuple[VideoID, ImageID, ImagePath]
VideoInfo = Tuple[VideoID, ImageID, Path, int, float, int, float]

# Type variables for generic typing
T = TypeVar('T')
R = TypeVar('R')


class ExtractorError(Exception):
    """Base exception for all extractor-related errors."""
    pass


class InputParsingError(ExtractorError):
    """Raised when there's an issue parsing input files or directories."""
    pass


class ExtractionError(ExtractorError):
    """Raised when feature extraction fails."""
    pass


class BatchProcessingError(ExtractorError):
    """Raised when batch processing fails."""
    pass


class ResourceAllocationError(ExtractorError):
    """Raised when resource allocation (e.g., GPU memory) fails."""
    pass


class ProcessingMode(Enum):
    """Enum for different processing modes."""
    SINGLE = "single"
    BATCH = "batch"
    STREAM = "stream"
    ASYNC = "async"


@dataclass
class ProgressTracker:
    """Progress tracker for long-running operations with CLI and async support."""
    
    initial: int = 0
    total: int = 0
    description: str = "Processing"
    unit: str = "it"
    disable: bool = False
    _pbar: Optional[tqdm] = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize the progress bar."""
        self._pbar = tqdm(
            initial=self.initial,
            total=self.total,
            desc=self.description,
            unit=self.unit,
            disable=self.disable
        )
    
    def update(self, n: int = 1) -> None:
        """Update the progress bar.
        
        Args:
            n: Number of iterations to increment
        """
        if self._pbar:
            self._pbar.update(n)
    
    def set_description(self, desc: str) -> None:
        """Set the description of the progress bar.
        
        Args:
            desc: New description
        """
        if self._pbar:
            self._pbar.set_description(desc)
    
    def set_postfix(self, **kwargs: Any) -> None:
        """Set the postfix of the progress bar.
        
        Args:
            **kwargs: Key-value pairs to display
        """
        if self._pbar:
            self._pbar.set_postfix(**kwargs)
    
    def close(self) -> None:
        """Close the progress bar."""
        if self._pbar:
            self._pbar.close()
            self._pbar = None
    
    def __call__(self, iterable: Iterable[T]) -> Iterator[T]:
        """Wrap an iterable with the progress bar.
        
        Args:
            iterable: The iterable to wrap
            
        Returns:
            A wrapped iterable that updates the progress bar
        """
        if self._pbar:
            return self._pbar(iterable)
        return iter(iterable)
    
    async def track_async(self, iterable: AsyncIterable[T]) -> AsyncGenerator[T, None]:
        """Track progress of an async iterable.
        
        Args:
            iterable: The async iterable to track
            
        Yields:
            Items from the async iterable
        """
        async for item in async_tqdm(
            iterable,
            initial=self.initial,
            total=self.total,
            desc=self.description,
            unit=self.unit,
            disable=self.disable
        ):
            yield item


# Backward compatibility
CliProgress = ProgressTracker


class BaseExtractor(ABC):
    """Base class for all extractors.
    
    This abstract class provides the foundation for implementing feature extractors
    that process images. It handles input parsing, batch processing, and result saving.
    """
    
    @classmethod
    def add_arguments(cls, parser: Any) -> None:
        """Add arguments to the parser.
        
        Args:
            parser: The argument parser to add arguments to
        """
        parser.add_argument('--chunk-size', type=int, default=-1, 
                           help='if the extractor does not support streaming extraction, '
                                'send this many image paths to the extractor at once. '
                                '-1 means send all at once.')
        parser.add_argument('--force', default=False, action='store_true', 
                           help='overwrite existing data')
        parser.add_argument('--gpu', default=False, action='store_true', 
                           help='use the GPU if available')
        parser.add_argument('--save-every', type=int, default=5000, 
                           help='flush every N records extracted')
        parser.add_argument('--batch-size', type=int, default=32,
                           help='batch size for GPU processing')
        parser.add_argument('--num-workers', type=int, default=4,
                           help='number of worker processes/threads')
        parser.add_argument('--cache-size', type=int, default=1024,
                           help='size of the LRU cache for repeated operations')
        parser.add_argument('--memory-limit', type=float, default=0.8,
                           help='fraction of GPU memory to use (0.0-1.0)')
        parser.add_argument('--processing-mode', type=str, 
                           choices=[mode.value for mode in ProcessingMode], 
                           default=ProcessingMode.BATCH.value,
                           help='processing mode: single, batch, stream, or async')

        parser.add_argument('input_images', type=Path, 
                           help='images to be processed. '
                                'Can be a directory or a file with a list of images. '
                                'Each line of the list must be in the format: '
                                '[[<video_id>\\t]<image_id>\\t]<image_path>\\n '
                                'If <video_id> is specified, contiguos images with the same '
                                '<video_id> will be grouped together in the output files. '
                                'If <image_id> is not specified, an incremental number '
                                'will be used instead.')

        subparsers = parser.add_subparsers(dest='output_type')

        file_parser = subparsers.add_parser('jsonl', help='save results to gzipped JSONL files')
        file_parser.add_argument('-o', '--output', type=Path, 
                               help='output path template, where "{video_id}" '
                                    'will be replaced by the video id.')

        hdf5_parser = subparsers.add_parser('hdf5', help='save results to HDF5 files')
        hdf5_parser.add_argument('-n', '--features-name', default='generic', 
                               help='identifier of feature type')
        hdf5_parser.add_argument('-o', '--output', type=Path, 
                               help='output path template, where "{video_id}" '
                                    'will be replaced by the video id.')

    def __init__(self, args: Any) -> None:
        """Initialize the extractor.
        
        Args:
            args: The parsed command-line arguments
        """
        super().__init__()
        self.args = args
        self._setup_processing_mode()
        self._setup_gpu()
        self._setup_caching()
    
    def _setup_processing_mode(self) -> None:
        """Set up the processing mode based on arguments."""
        mode_str = getattr(self.args, 'processing_mode', ProcessingMode.BATCH.value)
        self.processing_mode = ProcessingMode(mode_str)
        logger.info(f"Processing mode: {self.processing_mode.value}")
    
    def _setup_gpu(self) -> None:
        """Set up GPU usage based on arguments."""
        self.use_gpu = getattr(self.args, 'gpu', False)
        if self.use_gpu and torch.cuda.is_available():
            # Set memory limit if specified
            memory_limit = getattr(self.args, 'memory_limit', 0.8)
            if 0.0 < memory_limit <= 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(memory_limit)
                    logger.info(f"GPU memory limit set to {memory_limit*100:.0f}%")
                except Exception as e:
                    logger.warning(f"Failed to set GPU memory limit: {e}")
            
            # Enable cudnn benchmark for faster training
            torch.backends.cudnn.benchmark = True
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif self.use_gpu:
            logger.warning("GPU requested but not available, falling back to CPU")
            self.use_gpu = False
        else:
            logger.info("Using CPU for processing")
    
    def _setup_caching(self) -> None:
        """Set up caching based on arguments."""
        self.cache_size = getattr(self.args, 'cache_size', 1024)
        logger.info(f"LRU cache size: {self.cache_size}")

    def parse_input(self) -> List[ImageInfo]:
        """Parse the input file and return a list of (video_id, frame_id, frame_path) tuples.
        
        Returns:
            List of tuples containing video ID, frame ID, and frame path
            
        Raises:
            InputParsingError: If there's an issue parsing the input
        """
        input_path = self.args.input_images
        
        try:
            if input_path.is_dir():  # input is a directory, list all images in it
                image_paths = sorted(input_path.glob("*.png")) or sorted(input_path.glob("*.jpg"))
                if not image_paths:
                    raise InputParsingError(f"No PNG or JPG images found in directory: {input_path}")
                ids_and_paths = [(input_path.name, p.stem, p) for p in image_paths]
            else:  # input is a file, parse it
                with input_path.open() as image_list:
                    reader = csv.reader(image_list, delimiter='\t')

                    # peek at the first line to determine the number of columns
                    try:
                        peek, reader = itertools.tee(reader)
                        row = next(peek, None)
                        if row is None:
                            raise InputParsingError(f"Empty input file: {input_path}")
                        num_cols = len(row)
                    except StopIteration:
                        raise InputParsingError(f"Failed to read first line from: {input_path}")

                    # parse the rest of the file
                    if num_cols == 1:
                        parse_row = lambda row: ('', str(row[0]), Path(row[0]))
                    elif num_cols == 2:
                        parse_row = lambda row: ('', row[0], Path(row[1]))
                    elif num_cols == 3:
                        parse_row = lambda row: (row[0], row[1], Path(row[2]))
                    else:
                        raise InputParsingError(
                            f"Unexpected number of columns ({num_cols}) in input file: {input_path}"
                        )

                    ids_and_paths = [parse_row(row) for row in reader]
        except (IOError, OSError) as e:
            raise InputParsingError(f"Failed to read input file or directory: {e}")
        
        if not ids_and_paths:
            raise InputParsingError(f"No valid image paths found in: {input_path}")
            
        return ids_and_paths

    def get_saver(self, video_id: str) -> AbstractSaver:
        """Return a saver for the given video id.
        
        Args:
            video_id: The video ID to get a saver for
            
        Returns:
            A saver instance for the specified video ID
            
        Raises:
            ValueError: If the output type is not supported
        """
        output_path = str(self.args.output).format(video_id=video_id)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_every = self.args.save_every

        if self.args.output_type == 'jsonl':
            return GzipJsonlFile(output_path, flush_every=save_every)
        elif self.args.output_type == 'hdf5':
            return HDF5File(output_path, flush_every=save_every, 
                          attrs={'features_name': self.args.features_name})
        else:
            raise ValueError(f"Unsupported output type: {self.args.output_type}")

    @abstractmethod
    def extract(self, image_paths: List[ImagePath]) -> List[RecordType]:
        """Load a batch of images and extract features.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            List of extracted feature records
            
        Raises:
            ExtractionError: If feature extraction fails
        """
        raise NotImplementedError()
    
    async def extract_async(self, image_paths: List[ImagePath]) -> List[RecordType]:
        """Extract features asynchronously.
        
        Default implementation runs the synchronous extract method in a thread pool.
        Subclasses can override this for true asynchronous processing.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            List of extracted feature records
        """
        return await asyncio.to_thread(self.extract, image_paths)

    def extract_iterable(self, image_paths: Iterable[ImagePath]) -> Iterable[RecordType]:
        """Consume an iterable and return an iterable of records.
        
        This method contains a fallback implementation using chunked processing,
        but subclasses can implement optimized solutions here.
        
        Args:
            image_paths: Iterable of image paths
            
        Returns:
            Iterable of extracted feature records
            
        Raises:
            BatchProcessingError: If batch processing fails
        """
        try:
            if self.processing_mode == ProcessingMode.STREAM:
                # Process one by one for memory efficiency
                for image_path in image_paths:
                    yield from self.extract([image_path])
            else:
                # Use batch processing
                batch_size = self.args.batch_size if self.processing_mode == ProcessingMode.BATCH else 1
                
                if self.args.chunk_size > 0:
                    batched_image_paths = more_itertools.chunked(image_paths, self.args.chunk_size)
                else:
                    warnings.warn(
                        'Using chunked processing with chunk_size=-1. '
                        'This may cause memory issues and progress not showing correctly. '
                        'Set a positive chunk_size or implement extract_iterable() in your extractor to avoid this.'
                    )
                    batched_image_paths = [list(image_paths)]
                
                # Process batches
                for batch in batched_image_paths:
                    # Further divide into mini-batches for GPU processing if needed
                    if self.use_gpu and len(batch) > batch_size:
                        for mini_batch in more_itertools.chunked(batch, batch_size):
                            yield from self.extract(mini_batch)
                    else:
                        yield from self.extract(batch)
        except Exception as e:
            raise BatchProcessingError(f"Failed during batch processing: {e}")

    async def extract_iterable_async(
        self, image_paths: AsyncIterable[ImagePath]
    ) -> AsyncGenerator[RecordType, None]:
        """Process an async iterable of image paths and yield records asynchronously.
        
        Args:
            image_paths: Async iterable of image paths
            
        Yields:
            Extracted feature records
            
        Raises:
            BatchProcessingError: If batch processing fails
        """
        try:
            batch_size = self.args.batch_size
            batch: List[ImagePath] = []
            
            async for image_path in image_paths:
                batch.append(image_path)
                
                if len(batch) >= batch_size:
                    for record in await self.extract_async(batch):
                        yield record
                    batch = []
            
            # Process remaining items
            if batch:
                for record in await self.extract_async(batch):
                    yield record
        except Exception as e:
            raise BatchProcessingError(f"Failed during async batch processing: {e}")

    def skip_existing(
        self, ids_and_paths: List[ImageInfo], progress: Optional[ProgressTracker] = None
    ) -> List[ImageInfo]:
        """Skip images that have already been processed.
        
        Args:
            ids_and_paths: List of (video_id, image_id, image_path) tuples
            progress: Optional progress tracker to update
            
        Returns:
            Filtered list of tuples, excluding already processed items
            
        Raises:
            SaverError: If there's an issue accessing the saved data
        """
        result: List[ImageInfo] = []
        
        try:
            for video_id, group in itertools.groupby(ids_and_paths, key=lambda x: x[0]):
                group_list = list(group)
                with self.get_saver(video_id) as saver:
                    to_be_processed = []
                    for video_id, image_id, image_path in group_list:
                        if image_id not in saver:
                            to_be_processed.append((video_id, image_id, image_path))
                        elif progress:
                            progress.update(1)
                result.extend(to_be_processed)
        except SaverError as e:
            logger.error(f"Error while checking for existing records: {e}")
            # Fall back to processing all items
            result = ids_and_paths
            
        return result

    async def skip_existing_async(
        self, ids_and_paths: List[ImageInfo], progress: Optional[ProgressTracker] = None
    ) -> List[ImageInfo]:
        """Skip images that have already been processed (async version).
        
        Args:
            ids_and_paths: List of (video_id, image_id, image_path) tuples
            progress: Optional progress tracker to update
            
        Returns:
            Filtered list of tuples, excluding already processed items
        """
        return await asyncio.to_thread(self.skip_existing, ids_and_paths, progress)

    @functools.lru_cache(maxsize=128)
    def cached_extract_single(self, image_path: str) -> RecordType:
        """Extract features for a single image with caching.
        
        This method caches results for frequently accessed images.
        
        Args:
            image_path: Path to the image as a string (for hashability)
            
        Returns:
            Extracted feature record
        """
        path_obj = Path(image_path)
        records = self.extract([path_obj])
        if not records:
            raise ExtractionError(f"No features extracted for {image_path}")
        return records[0]

    def run(self) -> None:
        """Run the extraction process.
        
        This method orchestrates the entire extraction pipeline:
        1. Parse input files
        2. Skip already processed items
        3. Extract features in batches
        4. Save results
        
        Raises:
            ExtractorError: If any part of the extraction process fails
        """
        try:
            # Parse input
            ids_and_paths = self.parse_input()
            n_images = len(ids_and_paths)
            logger.info(f"Found {n_images} images to process")

            # Set up progress tracking
            progress = ProgressTracker(
                initial=0, 
                total=n_images, 
                description="Extracting features",
                unit="img"
            )

            # Skip existing items if not forcing reprocessing
            if not self.args.force:
                ids_and_paths = self.skip_existing(ids_and_paths, progress)
                logger.info(f"{len(ids_and_paths)} images need processing")

            # Check if there's anything to process
            if not ids_and_paths:
                logger.info("All images have already been processed")
                progress.close()
                return

            # Unzip ids and paths
            video_ids, image_ids, image_paths = zip(*ids_and_paths)

            # Process images based on selected mode
            if self.processing_mode == ProcessingMode.ASYNC:
                asyncio.run(self._run_async(video_ids, image_ids, image_paths, progress))
            else:
                self._run_sync(video_ids, image_ids, image_paths, progress)
                
            progress.close()
            logger.info("Extraction completed successfully")
            
        except ExtractorError as e:
            logger.error(f"Extraction failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {e}", exc_info=True)
            raise ExtractorError(f"Unexpected error: {e}")

    def _run_sync(
        self, 
        video_ids: Sequence[str], 
        image_ids: Sequence[str], 
        image_paths: Sequence[ImagePath],
        progress: ProgressTracker
    ) -> None:
        """Run the extraction process synchronously.
        
        Args:
            video_ids: Sequence of video IDs
            image_ids: Sequence of image IDs
            image_paths: Sequence of image paths
            progress: Progress tracker
        """
        # Process images in batches
        records = self.extract_iterable(image_paths)
        ids_and_records = zip(video_ids, image_ids, records)
        ids_and_records = progress(ids_and_records)

        # Group images by video id and save
        for video_id, group in itertools.groupby(ids_and_records, key=lambda x: x[0]):
            records = [{'_id': _id, **record} for _, _id, record in group]
            with self.get_saver(video_id) as saver:
                saver.add_many(records, force=self.args.force)

    async def _run_async(
        self, 
        video_ids: Sequence[str], 
        image_ids: Sequence[str], 
        image_paths: Sequence[ImagePath],
        progress: ProgressTracker
    ) -> None:
        """Run the extraction process asynchronously.
        
        Args:
            video_ids: Sequence of video IDs
            image_ids: Sequence of image IDs
            image_paths: Sequence of image paths
            progress: Progress tracker
        """
        # Create async generators
        async def path_generator() -> AsyncGenerator[ImagePath, None]:
            for path in image_paths:
                yield path
                
        # Process images asynchronously
        records_gen = self.extract_iterable_async(path_generator())
        
        # Group by video_id and save
        buffer: Dict[str, List[Tuple[str, RecordType]]] = {}
        buffer_size = self.args.save_every
        
        # Process records as they become available
        i = 0
        async for record in progress.track_async(records_gen):
            video_id = video_ids[i]
            image_id = image_ids[i]
            
            if video_id not in buffer:
                buffer[video_id] = []
            
            buffer[video_id].append((image_id, record))
            
            # Flush buffer when it gets too large
            if len(buffer[video_id]) >= buffer_size:
                await self._flush_buffer_async(video_id, buffer)
            
            i += 1
        
        # Flush remaining buffers
        for video_id in list(buffer.keys()):
            if buffer[video_id]:
                await self._flush_buffer_async(video_id, buffer)

    async def _flush_buffer_async(
        self, video_id: str, buffer: Dict[str, List[Tuple[str, RecordType]]]
    ) -> None:
        """Flush a buffer of records to storage asynchronously.
        
        Args:
            video_id: Video ID to flush
            buffer: Buffer of records by video ID
        """
        records = [{'_id': _id, **record} for _id, record in buffer[video_id]]
        buffer[video_id] = []
        
        with self.get_saver(video_id) as saver:
            await saver.add_many_async(records, force=self.args.force)


class BaseVideoExtractor(ABC):
    """Base class for all video extractors.
    
    This abstract class provides the foundation for implementing feature extractors
    that process videos. It handles input parsing, batch processing, and result saving.
    """
    
    @classmethod
    def add_arguments(cls, parser: Any) -> None:
        """Add arguments to the parser.
        
        Args:
            parser: The argument parser to add arguments to
        """
        parser.add_argument('--chunk-size', type=int, default=-1, 
                           help='if the extractor does not support streaming extraction, '
                                'send this many shot paths to the extractor at once. '
                                '-1 means send all at once.')
        parser.add_argument('--force', default=False, action='store_true', 
                           help='overwrite existing data')
        parser.add_argument('--gpu', default=False, action='store_true', 
                           help='use the GPU if available')
        parser.add_argument('--save-every', type=int, default=5000, 
                           help='flush every N records extracted')
        parser.add_argument('--batch-size', type=int, default=16,
                           help='batch size for GPU processing')
        parser.add_argument('--num-workers', type=int, default=4,
                           help='number of worker processes/threads')
        parser.add_argument('--cache-size', type=int, default=1024,
                           help='size of the LRU cache for repeated operations')
        parser.add_argument('--memory-limit', type=float, default=0.8,
                           help='fraction of GPU memory to use (0.0-1.0)')
        parser.add_argument('--processing-mode', type=str, 
                           choices=[mode.value for mode in ProcessingMode], 
                           default=ProcessingMode.BATCH.value,
                           help='processing mode: single, batch, stream, or async')
        parser.add_argument('--transcode', default=False, action='store_true',
                           help='transcode videos to a format optimized for processing')
        parser.add_argument('--transcode-format', type=str, default='mp4',
                           help='format to transcode videos to')
        parser.add_argument('--transcode-codec', type=str, default='h264',
                           help='codec to use for transcoding')
        parser.add_argument('--transcode-crf', type=int, default=23,
                           help='constant rate factor for transcoding (lower is better quality)')
        parser.add_argument('--transcode-preset', type=str, default='medium',
                           help='preset for transcoding (slower is better quality)')

        parser.add_argument('input_shots', type=Path, 
                           help='file containing the detected scenes. '
                                'It is a file containing scenes information, in the format '
                                '[Scene Number,Start Frame,Start Timecode,Start Time (seconds),'
                                'End Frame,End Timecode,End Time (seconds),Length (frames),'
                                'Length (timecode),Length (seconds)]')

        subparsers = parser.add_subparsers(dest='output_type')

        file_parser = subparsers.add_parser('jsonl', help='save results to gzipped JSONL files')
        file_parser.add_argument('-o', '--output', type=Path, 
                               help='output path template, where "{video_id}" '
                                    'will be replaced by the video id.')

        hdf5_parser = subparsers.add_parser('hdf5', help='save results to HDF5 files')
        hdf5_parser.add_argument('-n', '--features-name', default='generic', 
                               help='identifier of feature type')
        hdf5_parser.add_argument('-o', '--output', type=Path, 
                               help='output path template, where "{video_id}" '
                                    'will be replaced by the video id.')
        
        # Add report file argument
        parser.add_argument('--report', type=Path, default=None,
                           help='path to write a report about processed videos')

    def __init__(self, args: Any) -> None:
        """Initialize the extractor.
        
        Args:
            args: The parsed command-line arguments
        """
        super().__init__()
        self.args = args
        self._setup_processing_mode()
        self._setup_gpu()
        self._setup_caching()
        self._setup_reporting()
        
    def _setup_processing_mode(self) -> None:
        """Set up the processing mode based on arguments."""
        mode_str = getattr(self.args, 'processing_mode', ProcessingMode.BATCH.value)
        self.processing_mode = ProcessingMode(mode_str)
        logger.info(f"Processing mode: {self.processing_mode.value}")
    
    def _setup_gpu(self) -> None:
        """Set up GPU usage based on arguments."""
        self.use_gpu = getattr(self.args, 'gpu', False)
        if self.use_gpu and torch.cuda.is_available():
            # Set memory limit if specified
            memory_limit = getattr(self.args, 'memory_limit', 0.8)
            if 0.0 < memory_limit <= 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(memory_limit)
                    logger.info(f"GPU memory limit set to {memory_limit*100:.0f}%")
                except Exception as e:
                    logger.warning(f"Failed to set GPU memory limit: {e}")
            
            # Enable cudnn benchmark for faster training
            torch.backends.cudnn.benchmark = True
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif self.use_gpu:
            logger.warning("GPU requested but not available, falling back to CPU")
            self.use_gpu = False
        else:
            logger.info("Using CPU for processing")
    
    def _setup_caching(self) -> None:
        """Set up caching based on arguments."""
        self.cache_size = getattr(self.args, 'cache_size', 1024)
        logger.info(f"LRU cache size: {self.cache_size}")
    
    def _setup_reporting(self) -> None:
        """Set up reporting based on arguments."""
        self.report_path = getattr(self.args, 'report', None)
        self.failed_videos: Dict[str, str] = {}
        
    def _write_report(self) -> None:
        """Write a report about processed videos."""
        if not self.report_path:
            return
            
        try:
            with open(self.report_path, 'w') as f:
                f.write("# Video Processing Report\n\n")
                
                if self.failed_videos:
                    f.write("## Failed Videos\n\n")
                    f.write("| Video ID | Error |\n")
                    f.write("|----------|-------|\n")
                    for video_id, error in self.failed_videos.items():
                        f.write(f"| {video_id} | {error} |\n")
                else:
                    f.write("All videos processed successfully.\n")
        except (IOError, OSError) as e:
            logger.error(f"Failed to write report: {e}")

    def parse_input(self) -> List[VideoInfo]:
        """Parse the input file and return a list of
        (video_id, shot_id, video_path, start_frame, start_time, end_frame, end_time) tuples.
        
        Returns:
            List of tuples containing video information
            
        Raises:
            InputParsingError: If there's an issue parsing the input
        """
        input_shots = self.args.input_shots
        shot_frames: List[Tuple[str, str, str]] = []
        
        try:
            if input_shots.is_dir():
                # inputs_shots is the directory of a single video
                video_id = input_shots.stem
                frame_paths = sorted(input_shots.glob("*.png")) or sorted(input_shots.glob("*.jpg"))
                if not frame_paths:
                    raise InputParsingError(f"No PNG or JPG images found in directory: {input_shots}")
                frame_ids = [f.stem for f in frame_paths]
                shot_frames = list(zip(
                    itertools.repeat(video_id), 
                    frame_ids, 
                    [str(p) for p in frame_paths]
                ))
            else:
                # input_shots is a tsv file containing (video_id, frame_id, frame_path)
                with input_shots.open() as image_list:
                    shot_frames = list(csv.reader(image_list, delimiter='\t'))
                    
                    if not shot_frames:
                        raise InputParsingError(f"Empty input file: {input_shots}")
        except (IOError, OSError) as e:
            raise InputParsingError(f"Failed to read input file or directory: {e}")
        
        result: List[VideoInfo] = []
        
        # For each video, read the scenes.csv to get the time information
        for video_id, group in itertools.groupby(shot_frames, key=lambda x: x[0]):
            try:
                # Get scenes file and video path from the frame path
                group_list = list(group)
                frame_ids, frame_paths = zip(*[
                    (fid, Path(fpath)) for _, fid, fpath in group_list
                ])
                scenes_file = frame_paths[0].parent / f'{video_id}-scenes.csv'

                # Find the video file (it can have any extension)
                escaped_video_id = re.escape(video_id)
                candidates = (scenes_file.parents[2] / 'videos').glob(f'{video_id}.*')
                video_paths = [
                    c for c in candidates 
                    if re.match(rf'{escaped_video_id}\.[0-9a-zA-Z]+', c.name)
                ]
                
                if not video_paths:
                    error_msg = f"No video found for {video_id}"
                    self.failed_videos[video_id] = error_msg
                    logger.error(error_msg)
                    continue
                    
                video_path = video_paths[0]

                # Read the scenes.csv file
                with scenes_file.open() as scenes:
                    scenes_reader = csv.DictReader(scenes)
                    frame_id_to_timeinfo_dict = {
                        int(row['Scene Number']): (
                            int(row['Start Frame']),
                            float(row['Start Time (seconds)']),
                            int(row['End Frame']),
                            float(row['End Time (seconds)'])
                        ) for row in scenes_reader
                    }

                for frame_id, frame_path in zip(frame_ids, frame_paths):
                    try:
                        scene_id = int(re.split('-|_', frame_path.stem)[-1])
                        
                        if scene_id not in frame_id_to_timeinfo_dict:
                            logger.warning(f"Scene {scene_id} not found in {scenes_file}")
                            continue

                        timeinfo = frame_id_to_timeinfo_dict[scene_id]
                        row: VideoInfo = (video_id, frame_id, video_path, *timeinfo)
                        result.append(row)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse scene ID from {frame_path}: {e}")
                        continue
            except Exception as e:
                error_msg = f"Failed to process video {video_id}: {e}"
                self.failed_videos[video_id] = error_msg
                logger.error(error_msg)
                continue
                
        if not result:
            raise InputParsingError("No valid video shots found in the input")
            
        return result

    def get_saver(self, video_id: str, read_only: bool = False) -> AbstractSaver:
        """Return a saver for the given video id.
        
        Args:
            video_id: The video ID to get a saver for
            read_only: Whether to open the saver in read-only mode
            
        Returns:
            A saver instance for the specified video ID
            
        Raises:
            ValueError: If the output type is not supported
        """
        output_path = str(self.args.output).format(video_id=video_id)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_every = self.args.save_every

        if self.args.output_type == 'jsonl':
            return GzipJsonlFile(output_path, flush_every=save_every)
        elif self.args.output_type == 'hdf5':
            return HDF5File(
                output_path, 
                read_only=read_only, 
                flush_every=save_every, 
                attrs={'features_name': self.args.features_name}
            )
        else:
            raise ValueError(f"Unsupported output type: {self.args.output_type}")

    @abstractmethod
    def extract(self, shot_info: List[VideoInfo]) -> List[RecordType]:
        """Load a batch of shots and extract features.
        
        Args:
            shot_info: List of shot information tuples
            
        Returns:
            List of extracted feature records
            
        Raises:
            ExtractionError: If feature extraction fails
        """
        raise NotImplementedError()
    
    async def extract_async(self, shot_info: List[VideoInfo]) -> List[RecordType]:
        """Extract features asynchronously.
        
        Default implementation runs the synchronous extract method in a thread pool.
        Subclasses can override this for true asynchronous processing.
        
        Args:
            shot_info: List of shot information tuples
            
        Returns:
            List of extracted feature records
        """
        return await asyncio.to_thread(self.extract, shot_info)

    def extract_iterable(self, shot_infos: Iterable[VideoInfo]) -> Iterable[RecordType]:
        """Consume an iterable and return an iterable of records.
        
        This method contains a fallback implementation using chunked processing,
        but subclasses can implement optimized solutions here.
        
        Args:
            shot_infos: Iterable of shot information tuples
            
        Returns:
            Iterable of extracted feature records
            
        Raises:
            BatchProcessingError: If batch processing fails
        """
        try:
            if self.processing_mode == ProcessingMode.STREAM:
                # Process one by one for memory efficiency
                for shot_info in shot_infos:
                    yield from self.extract([shot_info])
            else:
                # Use batch processing
                batch_size = self.args.batch_size if self.processing_mode == ProcessingMode.BATCH else 1
                
                if self.args.chunk_size > 0:
                    batched_shot_infos = more_itertools.chunked(shot_infos, self.args.chunk_size)
                else:
                    warnings.warn(
                        'Using chunked processing with chunk_size=-1. '
                        'This may cause memory issues and progress not showing correctly. '
                        'Set a positive chunk_size or implement extract_iterable() in your extractor to avoid this.'
                    )
                    batched_shot_infos = [list(shot_infos)]
                
                # Process batches
                for batch in batched_shot_infos:
                    # Further divide into mini-batches for GPU processing if needed
                    if self.use_gpu and len(batch) > batch_size:
                        for mini_batch in more_itertools.chunked(batch, batch_size):
                            yield from self.extract(mini_batch)
                    else:
                        yield from self.extract(batch)
        except Exception as e:
            raise BatchProcessingError(f"Failed during batch processing: {e}")

    async def extract_iterable_async(
        self, shot_infos: AsyncIterable[VideoInfo]
    ) -> AsyncGenerator[RecordType, None]:
        """Process an async iterable of shot information and yield records asynchronously.
        
        Args:
            shot_infos: Async iterable of shot information tuples
            
        Yields:
            Extracted feature records
            
        Raises:
            BatchProcessingError: If batch processing fails
        """
        try:
            batch_size = self.args.batch_size
            batch: List[VideoInfo] = []
            
            async for shot_info in shot_infos:
                batch.append(shot_info)
                
                if len(batch) >= batch_size:
                    for record in await self.extract_async(batch):
                        yield record
                    batch = []
            
            # Process remaining items
            if batch:
                for record in await self.extract_async(batch):
                    yield record
        except Exception as e:
            raise BatchProcessingError(f"Failed during async batch processing: {e}")

    def skip_existing(
        self, shot_paths_and_times: List[VideoInfo], progress: Optional[ProgressTracker] = None
    ) -> List[VideoInfo]:
        """Skip shots that have already been processed.
        
        Args:
            shot_paths_and_times: List of shot information tuples
            progress: Optional progress tracker to update
            
        Returns:
            Filtered list of tuples, excluding already processed items
            
        Raises:
            SaverError: If there's an issue accessing the saved data
        """
        result: List[VideoInfo] = []
        
        try:
            for video_id, group in itertools.groupby(shot_paths_and_times, key=lambda x: x[0]):
                group_list = list(group)
                with self.get_saver(video_id, read_only=True) as saver:
                    to_be_processed = []
                    for video_id, shot_id, *other in group_list:
                        if shot_id not in saver:
                            to_be_processed.append((video_id, shot_id, *other))
                        elif progress:
                            progress.update(1)
                result.extend(to_be_processed)
        except SaverError as e:
            logger.error(f"Error while checking for existing records: {e}")
            # Fall back to processing all items
            result = shot_paths_and_times
            
        return result

    async def skip_existing_async(
        self, shot_paths_and_times: List[VideoInfo], progress: Optional[ProgressTracker] = None
    ) -> List[VideoInfo]:
        """Skip shots that have already been processed (async version).
        
        Args:
            shot_paths_and_times: List of shot information tuples
            progress: Optional progress tracker to update
            
        Returns:
            Filtered list of tuples, excluding already processed items
        """
        return await asyncio.to_thread(self.skip_existing, shot_paths_and_times, progress)

    @functools.lru_cache(maxsize=128)
    def cached_extract_single(self, shot_info_str: str) -> RecordType:
        """Extract features for a single shot with caching.
        
        This method caches results for frequently accessed shots.
        
        Args:
            shot_info_str: Shot information as a string (for hashability)
            
        Returns:
            Extracted feature record
            
        Raises:
            ExtractionError: If feature extraction fails
        """
        # Parse the shot info string back into a tuple
        parts = shot_info_str.split('|')
        if len(parts) < 7:
            raise ValueError(f"Invalid shot info string: {shot_info_str}")
            
        video_id = parts[0]
        shot_id = parts[1]
        video_path = Path(parts[2])
        start_frame = int(parts[3])
        start_time = float(parts[4])
        end_frame = int(parts[5])
        end_time = float(parts[6])
        
        shot_info: VideoInfo = (video_id, shot_id, video_path, start_frame, start_time, end_frame, end_time)
        records = self.extract([shot_info])
        if not records:
            raise ExtractionError(f"No features extracted for {shot_info_str}")
        return records[0]
    
    def _maybe_transcode_video(self, video_path: Path) -> Path:
        """Transcode a video to a format optimized for processing if needed.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the transcoded video or the original if transcoding is disabled
            
        Raises:
            RuntimeError: If transcoding fails
        """
        if not getattr(self.args, 'transcode', False):
            return video_path
            
        try:
            import subprocess
            from tempfile import gettempdir
            
            # Create a temporary file for the transcoded video
            temp_dir = Path(gettempdir())
            output_path = temp_dir / f"{video_path.stem}_transcoded.{self.args.transcode_format}"
            
            # Skip if already transcoded
            if output_path.exists():
                logger.info(f"Using existing transcoded video: {output_path}")
                return output_path
                
            logger.info(f"Transcoding {video_path} to {output_path}")
            
            # Build the ffmpeg command
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i", str(video_path),
                "-c:v", self.args.transcode_codec,
                "-crf", str(self.args.transcode_crf),
                "-preset", self.args.transcode_preset,
                "-c:a", "aac",  # Audio codec
                str(output_path)
            ]
            
            # Run the command
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                error_msg = f"Transcoding failed: {result.stderr}"
                logger.error(error_msg)
                self.failed_videos[video_path.stem] = f"Transcoding failed: {result.returncode}"
                return video_path
                
            logger.info(f"Transcoded {video_path} to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Transcoding error: {e}")
            self.failed_videos[video_path.stem] = f"Transcoding error: {str(e)}"
            return video_path

    def run(self) -> None:
        """Run the extraction process.
        
        This method orchestrates the entire extraction pipeline:
        1. Parse input files
        2. Skip already processed items
        3. Extract features in batches
        4. Save results
        
        Raises:
            ExtractorError: If any part of the extraction process fails
        """
        try:
            # Parse input
            shot_paths_and_times = self.parse_input()
            n_shots = len(shot_paths_and_times)
            logger.info(f"Found {n_shots} shots to process")

            # Set up progress tracking
            progress = ProgressTracker(
                initial=0, 
                total=n_shots, 
                description="Extracting features",
                unit="shot"
            )

            # Skip existing items if not forcing reprocessing
            if not self.args.force:
                shot_paths_and_times = self.skip_existing(shot_paths_and_times, progress)
                logger.info(f"{len(shot_paths_and_times)} shots need processing")

            # Check if there's anything to process
            if not shot_paths_and_times:
                logger.info("All shots have already been processed")
                progress.close()
                self._write_report()
                return

            # Process shots based on selected mode
            if self.processing_mode == ProcessingMode.ASYNC:
                asyncio.run(self._run_async(shot_paths_and_times, progress))
            else:
                self._run_sync(shot_paths_and_times, progress)
                
            progress.close()
            self._write_report()
            logger.info("Extraction completed successfully")
            
        except ExtractorError as e:
            logger.error(f"Extraction failed: {e}")
            self._write_report()
            raise
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {e}", exc_info=True)
            self._write_report()
            raise ExtractorError(f"Unexpected error: {e}")

    def _run_sync(self, shot_paths_and_times: List[VideoInfo], progress: ProgressTracker) -> None:
        """Run the extraction process synchronously.
        
        Args:
            shot_paths_and_times: List of shot information tuples
            progress: Progress tracker
        """
        # Process shots in batches
        records = self.extract_iterable(shot_paths_and_times)
        
        # Unzip shot paths and times
        shot_paths_and_times_unzipped = more_itertools.unzip(shot_paths_and_times)
        shot_paths_and_times_unzipped = more_itertools.padded(
            shot_paths_and_times_unzipped, fillvalue=(), n=9
        )
        video_ids, image_ids, *_ = shot_paths_and_times_unzipped
        
        shot_ids_and_records = zip(video_ids, image_ids, records)
        shot_ids_and_records = progress(shot_ids_and_records)

        # Group shots by video id and save
        for video_id, group in itertools.groupby(shot_ids_and_records, key=lambda x: x[0]):
            records = [{'_id': _id, **record} for _, _id, record in group]
            with self.get_saver(video_id) as saver:
                saver.add_many(records, force=self.args.force)

    async def _run_async(self, shot_paths_and_times: List[VideoInfo], progress: ProgressTracker) -> None:
        """Run the extraction process asynchronously.
        
        Args:
            shot_paths_and_times: List of shot information tuples
            progress: Progress tracker
        """
        # Create async generators
        async def shot_generator() -> AsyncGenerator[VideoInfo, None]:
            for shot in shot_paths_and_times:
                yield shot
                
        # Process shots asynchronously
        records_gen = self.extract_iterable_async(shot_generator())
        
        # Group by video_id and save
        buffer: Dict[str, List[Tuple[str, RecordType]]] = {}
        buffer_size = self.args.save_every
        
        # Process records as they become available
        i = 0
        async for record in progress.track_async(records_gen):
            video_id = shot_paths_and_times[i][0]
            shot_id = shot_paths_and_times[i][1]
            
            if video_id not in buffer:
                buffer[video_id] = []
            
            buffer[video_id].append((shot_id, record))
            
            # Flush buffer when it gets too large
            if len(buffer[video_id]) >= buffer_size:
                await self._flush_buffer_async(video_id, buffer)
            
            i += 1
        
        # Flush remaining buffers
        for video_id in list(buffer.keys()):
            if buffer[video_id]:
                await self._flush_buffer_async(video_id, buffer)

    async def _flush_buffer_async(
        self, video_id: str, buffer: Dict[str, List[Tuple[str, RecordType]]]
    ) -> None:
        """Flush a buffer of records to storage asynchronously.
        
        Args:
            video_id: Video ID to flush
            buffer: Buffer of records by video ID
        """
        records = [{'_id': _id, **record} for _id, record in buffer[video_id]]
        buffer[video_id] = []
        
        with self.get_saver(video_id) as saver:
            await saver.add_many_async(records, force=self.args.force)
