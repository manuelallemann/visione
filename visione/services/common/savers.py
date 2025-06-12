"""
Savers module for persisting extracted data in various formats.
Provides abstract base class and concrete implementations for different storage formats.
"""
import asyncio
import functools
import gzip
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar, Union, cast

import h5py
import numpy as np

# Type variables for generic typing
T = TypeVar('T')
RecordType = Dict[str, Any]
PathLike = Union[str, Path]


class SaverError(Exception):
    """Base exception for all saver-related errors."""
    pass


class FileAccessError(SaverError):
    """Raised when there's an issue accessing the file."""
    pass


class DataValidationError(SaverError):
    """Raised when data validation fails."""
    pass


class AbstractSaver(ABC, Generic[T]):
    """Abstract base class for all data savers."""
    
    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the saved data.
        
        Args:
            key: The key to check
            
        Returns:
            True if the key exists, False otherwise
        """
        pass
    
    @abstractmethod
    def add(self, record: T, force: bool = False) -> None:
        """Add a single record to the saver.
        
        Args:
            record: The record to add
            force: If True, overwrite existing records with the same ID
            
        Raises:
            DataValidationError: If the record doesn't have an _id field and force is False
        """
        pass
    
    @abstractmethod
    def add_many(self, records: List[T], force: bool = False) -> None:
        """Add multiple records to the saver.
        
        Args:
            records: The records to add
            force: If True, overwrite existing records with the same ID
            
        Raises:
            DataValidationError: If any record doesn't have an _id field and force is False
        """
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush data to disk."""
        pass
    
    @abstractmethod
    async def add_async(self, record: T, force: bool = False) -> None:
        """Add a single record asynchronously.
        
        Args:
            record: The record to add
            force: If True, overwrite existing records with the same ID
            
        Raises:
            DataValidationError: If the record doesn't have an _id field and force is False
        """
        pass
    
    @abstractmethod
    async def add_many_async(self, records: List[T], force: bool = False) -> None:
        """Add multiple records asynchronously.
        
        Args:
            records: The records to add
            force: If True, overwrite existing records with the same ID
            
        Raises:
            DataValidationError: If any record doesn't have an _id field and force is False
        """
        pass


@dataclass
class GzipJsonlFile(AbstractSaver[RecordType]):
    """Saver implementation for gzipped JSONL files."""
    
    path: PathLike
    flush_every: int = 1000
    _file: Optional[gzip.GzipFile] = field(default=None, init=False, repr=False)
    _records_since_flush: int = field(default=0, init=False, repr=False)
    _existing_ids: Set[str] = field(default_factory=set, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize the saver after instance creation."""
        self.path = Path(self.path)
        self._load_existing_ids()
    
    def _load_existing_ids(self) -> None:
        """Load existing IDs from the file if it exists."""
        if not os.path.exists(self.path):
            return
            
        try:
            with gzip.open(self.path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if '_id' in record:
                            self._existing_ids.add(record['_id'])
                    except json.JSONDecodeError:
                        continue
        except (IOError, OSError) as e:
            raise FileAccessError(f"Failed to read existing IDs from {self.path}: {e}")
    
    def __enter__(self) -> 'GzipJsonlFile':
        """Context manager entry point."""
        try:
            self._file = gzip.open(self.path, 'at', encoding='utf-8')
            return self
        except (IOError, OSError) as e:
            raise FileAccessError(f"Failed to open file {self.path}: {e}")
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point."""
        if self._file:
            self._file.close()
            self._file = None
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the saved data."""
        return key in self._existing_ids
    
    @functools.lru_cache(maxsize=128)
    def _validate_record(self, record_id: str) -> None:
        """Validate a record ID and cache the result.
        
        Args:
            record_id: The record ID to validate
            
        Raises:
            DataValidationError: If the record ID is invalid
        """
        if not record_id:
            raise DataValidationError("Record ID cannot be empty")
    
    def add(self, record: RecordType, force: bool = False) -> None:
        """Add a single record to the file."""
        if not self._file:
            raise RuntimeError("File is not open. Use with statement.")
            
        if '_id' not in record:
            raise DataValidationError("Record must have an '_id' field")
            
        record_id = record['_id']
        self._validate_record(record_id)
            
        if not force and record_id in self._existing_ids:
            return
            
        try:
            self._file.write(json.dumps(record) + '\n')
            self._existing_ids.add(record_id)
            self._records_since_flush += 1
            
            if self._records_since_flush >= self.flush_every:
                self.flush()
        except (IOError, OSError) as e:
            raise FileAccessError(f"Failed to write record to {self.path}: {e}")
    
    def add_many(self, records: List[RecordType], force: bool = False) -> None:
        """Add multiple records to the file."""
        if not records:
            return
            
        for record in records:
            self.add(record, force)
    
    def flush(self) -> None:
        """Flush data to disk."""
        if self._file:
            self._file.flush()
            self._records_since_flush = 0
    
    async def add_async(self, record: RecordType, force: bool = False) -> None:
        """Add a single record asynchronously."""
        await asyncio.to_thread(self.add, record, force)
    
    async def add_many_async(self, records: List[RecordType], force: bool = False) -> None:
        """Add multiple records asynchronously."""
        if not records:
            return
            
        # Process in chunks to avoid blocking
        chunk_size = min(self.flush_every, 1000)
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i+chunk_size]
            await asyncio.to_thread(self.add_many, chunk, force)


@dataclass
class HDF5File(AbstractSaver[RecordType]):
    """Saver implementation for HDF5 files."""
    
    path: PathLike
    flush_every: int = 1000
    read_only: bool = False
    attrs: Dict[str, Any] = field(default_factory=dict)
    _file: Optional[h5py.File] = field(default=None, init=False, repr=False)
    _records_since_flush: int = field(default=0, init=False, repr=False)
    _existing_ids: Set[str] = field(default_factory=set, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize the saver after instance creation."""
        self.path = Path(self.path)
        self._load_existing_ids()
    
    def _load_existing_ids(self) -> None:
        """Load existing IDs from the file if it exists."""
        if not os.path.exists(self.path):
            return
            
        try:
            with h5py.File(self.path, 'r') as f:
                self._existing_ids = set(f.keys())
        except (IOError, OSError) as e:
            raise FileAccessError(f"Failed to read existing IDs from {self.path}: {e}")
    
    def __enter__(self) -> 'HDF5File':
        """Context manager entry point."""
        try:
            mode = 'r' if self.read_only else 'a'
            self._file = h5py.File(self.path, mode)
            
            # Set file attributes if not read-only
            if not self.read_only:
                for key, value in self.attrs.items():
                    self._file.attrs[key] = value
                    
            return self
        except (IOError, OSError) as e:
            raise FileAccessError(f"Failed to open file {self.path}: {e}")
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point."""
        if self._file:
            self._file.close()
            self._file = None
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the saved data."""
        return key in self._existing_ids
    
    def add(self, record: RecordType, force: bool = False) -> None:
        """Add a single record to the file."""
        if not self._file:
            raise RuntimeError("File is not open. Use with statement.")
            
        if self.read_only:
            raise RuntimeError("File is opened in read-only mode")
            
        if '_id' not in record:
            raise DataValidationError("Record must have an '_id' field")
            
        record_id = str(record['_id'])
        
        if not force and record_id in self._existing_ids:
            return
            
        try:
            # Remove the record if it exists and force is True
            if force and record_id in self._file:
                del self._file[record_id]
                
            # Create a group for the record
            group = self._file.create_group(record_id)
            
            # Add all fields to the group
            for key, value in record.items():
                if key == '_id':
                    continue
                    
                # Handle different types of data
                if isinstance(value, (list, np.ndarray)):
                    value_array = np.array(value)
                    group.create_dataset(key, data=value_array, compression="gzip")
                elif isinstance(value, (int, float, str, bool)):
                    group.attrs[key] = value
                else:
                    # For complex types, store as JSON
                    group.attrs[key] = json.dumps(value)
            
            self._existing_ids.add(record_id)
            self._records_since_flush += 1
            
            if self._records_since_flush >= self.flush_every:
                self.flush()
        except (IOError, OSError) as e:
            raise FileAccessError(f"Failed to write record to {self.path}: {e}")
    
    def add_many(self, records: List[RecordType], force: bool = False) -> None:
        """Add multiple records to the file."""
        if not records:
            return
            
        for record in records:
            self.add(record, force)
    
    def flush(self) -> None:
        """Flush data to disk."""
        if self._file:
            self._file.flush()
            self._records_since_flush = 0
    
    async def add_async(self, record: RecordType, force: bool = False) -> None:
        """Add a single record asynchronously."""
        await asyncio.to_thread(self.add, record, force)
    
    async def add_many_async(self, records: List[RecordType], force: bool = False) -> None:
        """Add multiple records asynchronously."""
        if not records:
            return
            
        # Process in chunks to avoid blocking
        chunk_size = min(self.flush_every, 1000)
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i+chunk_size]
            await asyncio.to_thread(self.add_many, chunk, force)
            
    @functools.lru_cache(maxsize=128)
    def get_cached(self, record_id: str, field_name: str) -> Any:
        """Get a field from a record with caching.
        
        Args:
            record_id: The ID of the record
            field_name: The name of the field to get
            
        Returns:
            The field value
            
        Raises:
            KeyError: If the record or field doesn't exist
        """
        if not self._file:
            raise RuntimeError("File is not open. Use with statement.")
            
        if record_id not in self._file:
            raise KeyError(f"Record {record_id} not found")
            
        group = self._file[record_id]
        
        if field_name in group.attrs:
            value = group.attrs[field_name]
            # Try to parse JSON if it looks like it
            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
        elif field_name in group:
            return group[field_name][:]
        else:
            raise KeyError(f"Field {field_name} not found in record {record_id}")
