# Dataset Technical Design Document

This document describes the technical design and implementation of the data loading infrastructure in Relax, covering both the `Dataset` class (eager loading) and `StreamingDataset` class (lazy loading).

______________________________________________________________________

## 1. Overview

The Relax framework provides two dataset implementations to handle different scale and performance requirements:

| Class              | Loading Strategy                | Best For                                        |
| ------------------ | ------------------------------- | ----------------------------------------------- |
| `Dataset`          | Eager (all data loaded at init) | Small to medium datasets, fast random access    |
| `StreamingDataset` | Lazy (on-demand loading)        | Large datasets, memory-constrained environments |

Both classes inherit from `BaseDataset` and share a common interface, making them interchangeable through configuration.

______________________________________________________________________

## 2. Architecture

### 2.1 Class Hierarchy

```
BaseDataset (ABC)
├── Dataset            # Eager loading implementation
└── StreamingDataset   # Lazy loading implementation
```

### 2.2 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Data Pipeline Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌────────────────────┐    ┌─────────────────────┐      │
│  │  JSONL/      │    │  Dataset /         │    │  RolloutDataSource  │      │
│  │  Parquet     │───>│  StreamingDataset  │───>│  WithBuffer         │      │
│  │  Files       │    │                    │    │                     │      │
│  └──────────────┘    └────────────────────┘    └──────────┬──────────┘      │
│                                                           │                 │
│                                                           ▼                 │
│                      ┌─────────────────────────────────────────────────┐    │
│                      │           RolloutManager.generate()             │    │
│                      │   - get_samples(num_samples)                    │    │
│                      │   - Iterates until rollout_batch_size reached   │    │
│                      └─────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Module Structure

```
relax/utils/
├── data_utils.py           # Shared utilities: BaseDataset, read_file, process_raw_sample
├── data.py                 # Dataset (eager loading)
└── streaming_dataset.py    # StreamingDataset (lazy loading)

relax/engine/rollout/
└── data_source.py          # RolloutDataSource with factory function
```

______________________________________________________________________

## 3. BaseDataset

`BaseDataset` is an abstract base class that defines the common interface for all dataset implementations.

### 3.1 Interface

```python
class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index."""

    @abstractmethod
    def shuffle(self, epoch_id: int) -> None:
        """Shuffle the dataset for a new epoch."""
```

### 3.2 Common Parameters

| Parameter             | Type | Description                                     |
| --------------------- | ---- | ----------------------------------------------- |
| `tokenizer`           | Any  | Tokenizer for text processing and chat template |
| `processor`           | Any  | Processor for multimodal inputs                 |
| `max_length`          | int  | Maximum prompt length for filtering             |
| `prompt_key`          | str  | Key for prompt in data dictionary               |
| `multimodal_keys`     | dict | Mapping of multimodal types to data keys        |
| `label_key`           | str  | Key for labels in data                          |
| `tool_key`            | str  | Key for tools in data                           |
| `metadata_key`        | str  | Key for metadata in data                        |
| `system_prompt`       | str  | System prompt key or content                    |
| `seed`                | int  | Random seed for reproducible shuffling          |
| `apply_chat_template` | bool | Whether to apply chat template                  |
| `use_audio_in_video`  | bool | Whether to extract audio from video files       |

______________________________________________________________________

## 4. Dataset (Eager Loading)

### 4.1 Design Principles

The `Dataset` class loads all data into memory during initialization. This provides:

- **O(1) random access** after initialization
- **Consistent memory footprint** throughout training
- **Simple implementation** with straightforward shuffling

### 4.2 Implementation

```python
class Dataset(BaseDataset):
    def __init__(self, path, tokenizer, processor, max_length, ...):
        # Load all samples into memory
        origin_samples = []
        for data in read_file(path):
            sample = self._process_data(data)
            origin_samples.append(sample)

        # Apply length filtering at initialization
        self.origin_samples = filter_long_prompts(
            origin_samples, tokenizer, processor, max_length
        )
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id: int) -> None:
        if self.epoch_id == new_epoch_id:
            return
        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)
```

### 4.3 Use Cases

- **Small to medium datasets** (\< 1M samples)
- **Fast random access** requirements
- **Pre-filtered data** where length checking is acceptable at init

### 4.4 Limitations

- **High memory usage** for large datasets
- **Long initialization time** due to upfront loading and filtering
- **Inflexible** for datasets that don't fit in memory

______________________________________________________________________

## 5. StreamingDataset (Lazy Loading)

### 5.1 Design Principles

The `StreamingDataset` class loads data on-demand, providing:

- **Constant memory usage** regardless of dataset size
- **Fast initialization** (deferred loading)
- **LRU caching** for frequently accessed samples

### 5.2 Core Components

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       StreamingDataset Architecture                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐       │
│  │   IndexManager  │   │   SampleBuffer   │   │   StreamingReader  │       │
│  │  ─────────────  │   │  ──────────────  │   │  ────────────────  │       │
│  │  • Shuffle      │   │  • LRU cache     │   │  • File reading    │       │
│  │    permutation  │   │  • Hit tracking  │   │  • Random access   │       │
│  │  • Epoch        │   │  • Auto eviction │   │  • Line indexing   │       │
│  │    boundaries   │   │                  │   │                    │       │
│  └────────┬────────┘   └────────┬─────────┘   └──────────┬─────────┘       │
│           │                     │                        │                 │
│           └─────────────────────┼────────────────────────┘                 │
│                                 │                                          │
│                                 ▼                                          │
│                    ┌────────────────────────┐                              │
│                    │   StreamingDataset     │                              │
│                    │   ──────────────────   │                              │
│                    │   • get_batch(n)       │                              │
│                    │   • __len__            │                              │
│                    │   • shuffle(epoch_id)  │                              │
│                    │   • save/load state    │                              │
│                    └────────────────────────┘                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 5.2.1 StreamingReader

Handles file I/O with random access support:

```python
class StreamingReader:
    """File reader with random access via line offset indexing."""

    def __init__(self, path: str):
        self.path = path
        self._line_offsets = None  # Built lazily on first access
        self._total_lines = None

    def _build_index(self) -> None:
        """Scan file once to record byte offset of each line."""
        self._line_offsets = []
        with open(self.path, "rb") as f:
            offset = 0
            for line in f:
                self._line_offsets.append(offset)
                offset += len(line)
        self._total_lines = len(self._line_offsets)

    def __getitem__(self, idx: int) -> dict:
        """Read single line by index using offset."""
        offset = self._line_offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(offset)
            return json.loads(f.readline())
```

**Supported formats:**

- JSONL (`.jsonl`)
- Parquet (`.parquet`) - loaded into memory as Parquet is columnar

#### 5.2.2 SampleBuffer

LRU cache for processed samples:

```python
class SampleBuffer:
    """LRU cache to avoid re-processing samples."""

    def __init__(self, max_size: int = 10000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, idx: int) -> Optional[Sample]:
        if idx in self.cache:
            self.cache.move_to_end(idx)  # Mark as recently used
            self._hits += 1
            return self.cache[idx]
        self._misses += 1
        return None

    def put(self, idx: int, sample: Sample) -> None:
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Evict oldest
        self.cache[idx] = sample
```

#### 5.2.3 IndexManager

Manages shuffle permutation and epoch transitions:

```python
class IndexManager:
    """Reproducible shuffle with epoch boundary tracking."""

    def __init__(self, total_size: int, seed: int = 42):
        self.total_size = total_size
        self.seed = seed
        self.current_epoch = -1
        self.indices = None
        self.position = 0

    def shuffle(self, epoch_id: int) -> None:
        if epoch_id == self.current_epoch:
            return
        random.seed(self.seed + epoch_id)
        self.indices = list(range(self.total_size))
        random.shuffle(self.indices)
        self.current_epoch = epoch_id
        self.position = 0

    def get_next_indices(self, n: int) -> tuple[list[int], bool]:
        """Get next n indices, handling epoch boundary."""
        indices = []
        crossed_epoch = False
        while len(indices) < n:
            remaining = self.total_size - self.position
            if remaining <= 0:
                self.shuffle(self.current_epoch + 1)
                crossed_epoch = True
                remaining = self.total_size
            take = min(n - len(indices), remaining)
            indices.extend(self.indices[self.position:self.position + take])
            self.position += take
        return indices, crossed_epoch
```

### 5.3 Key Method: get_batch()

```python
def get_batch(self, n: int) -> tuple[list[Sample], bool]:
    """
    Get n valid samples, automatically handling filtering and epochs.

    Returns:
        (samples, crossed_epoch): List of samples and epoch boundary flag
    """
    samples = []
    crossed_epoch = False

    while len(samples) < n:
        indices, epoch_crossed = self.index_manager.get_next_indices(n - len(samples))
        crossed_epoch = crossed_epoch or epoch_crossed

        for idx in indices:
            # Check cache first
            sample = self.buffer.get(idx)
            if sample is None:
                raw_data = self.reader[idx]
                sample = self._process_raw_data(raw_data)
                if sample is not None:
                    self.buffer.put(idx, sample)

            if sample is not None:
                samples.append(sample)

    return samples, crossed_epoch
```

### 5.4 Use Cases

- **Large datasets** (> 1M samples or multimodal data)
- **Memory-constrained environments**
- **Scenarios where fast startup is important**

### 5.5 Configuration

| Parameter       | Default | Description                                           |
| --------------- | ------- | ----------------------------------------------------- |
| `buffer_size`   | 10000   | Maximum samples to cache in LRU buffer                |
| `prefetch_size` | 100     | Reserved for future prefetching (not yet implemented) |

______________________________________________________________________

## 6. Deferred Length Filtering

### 6.1 Problem

Traditional eager filtering requires tokenizing all samples at initialization:

```python
# Eager approach (Dataset)
self.origin_samples = filter_long_prompts(all_samples, tokenizer, processor, max_length)
```

This is expensive for large datasets.

### 6.2 Solution

`StreamingDataset` defers filtering to access time:

```python
def _process_raw_data(self, data: dict) -> Optional[Sample]:
    sample = self._process_data(data)

    # Check length at access time
    if self.max_length is not None:
        if not check_sample_length(sample, self.tokenizer, self.processor, self.max_length):
            self._filter_count += 1
            return None  # Filtered out

    return sample
```

**Trade-offs:**

- Faster initialization
- Filtering cost distributed over training
- Unknown final sample count until full traversal

______________________________________________________________________

## 7. Checkpoint and State Management

### 7.1 State Structure

```python
def get_state(self) -> dict:
    return {
        "epoch_id": self.index_manager.current_epoch,
        "position": self.index_manager.position,
        "filter_count": self._filter_count,
        "total_processed": self._total_processed,
    }

def load_state(self, state: dict) -> None:
    self.index_manager.load_state(state)
    self._filter_count = state.get("filter_count", 0)
    self._total_processed = state.get("total_processed", 0)
```

### 7.2 Distributed Training Considerations

- State is saved per-worker in `RolloutDataSource`
- Buffer contents are **not** saved (reconstructed on load)
- Epoch and position ensure reproducible resumption

______________________________________________________________________

## 8. Integration with RolloutDataSource

### 8.1 Factory Function

```python
def _create_dataset(args, tokenizer, processor):
    use_streaming = getattr(args, "use_streaming_dataset", False)

    if use_streaming:
        from relax.utils.data.streaming_dataset import StreamingDataset
        return StreamingDataset(
            path=args.prompt_data,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.rollout_max_prompt_len,
            buffer_size=getattr(args, "streaming_buffer_size", 10000),
            use_audio_in_video=getattr(args, "use_audio_in_video", False),
            # ... other parameters
        )
    else:
        return Dataset(
            args.prompt_data,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.rollout_max_prompt_len,
            use_audio_in_video=getattr(args, "use_audio_in_video", False),
            # ... other parameters
        )
```

### 8.2 Configuration Flags

| Argument                  | Type | Default | Description           |
| ------------------------- | ---- | ------- | --------------------- |
| `--use-streaming-dataset` | flag | False   | Enable streaming mode |
| `--streaming-buffer-size` | int  | 10000   | LRU buffer size       |

______________________________________________________________________

## 9. API Compatibility

| Method/Property     | Dataset | StreamingDataset | Notes                                             |
| ------------------- | ------- | ---------------- | ------------------------------------------------- |
| `__len__`           | ✅      | ✅               | StreamingDataset requires file scan on first call |
| `__getitem__`       | ✅      | ✅               | Both support random access                        |
| `shuffle(epoch_id)` | ✅      | ✅               | Fully compatible                                  |
| `samples` property  | ✅      | ✅\*             | StreamingDataset uses proxy for compatibility     |
| `get_batch(n)`      | ❌      | ✅               | StreamingDataset-specific batch API               |

\*StreamingDataset provides a `StreamingDatasetSamplesProxy` for backward compatibility with code expecting `dataset.samples[start:end]`.

______________________________________________________________________

## 10. Performance Characteristics

| Metric              | Dataset | StreamingDataset |
| ------------------- | ------- | ---------------- |
| Initialization Time | O(N)    | O(1)\*           |
| Memory Usage        | O(N)    | O(buffer_size)   |
| Random Access       | O(1)    | O(1)\*\*         |
| First `len()` call  | O(1)    | O(N)\*\*\*       |

\* Deferred until first access
\*\* After index is built
\*\*\* One-time file scan cost

______________________________________________________________________

## 11. Usage Examples

### 11.1 Using Dataset (Eager Loading)

```python
from relax.utils.data.data import Dataset

dataset = Dataset(
    path="data/train.jsonl",
    tokenizer=tokenizer,
    processor=processor,
    max_length=2048,
    prompt_key="text",
    apply_chat_template=True,
)

# Shuffle for new epoch
dataset.shuffle(epoch_id=1)

# Access samples
sample = dataset[0]
batch = dataset.samples[0:32]
```

### 11.2 Using StreamingDataset (Lazy Loading)

```python
from relax.utils.data.streaming_dataset import StreamingDataset

dataset = StreamingDataset(
    path="data/train.jsonl",
    tokenizer=tokenizer,
    processor=processor,
    max_length=2048,
    buffer_size=10000,
    prompt_key="text",
    apply_chat_template=True,
)

# Shuffle for new epoch
dataset.shuffle(epoch_id=1)

# Get batch (preferred API for streaming)
samples, crossed_epoch = dataset.get_batch(32)

# Random access also supported
sample = dataset[0]

# Get statistics
stats = dataset.get_stats()
print(f"Buffer hit rate: {stats['buffer_hit_rate']:.2%}")
```

______________________________________________________________________

## 12. Recommendations

| Scenario                   | Recommended Class | Reason                      |
| -------------------------- | ----------------- | --------------------------- |
| \< 100K samples            | Dataset           | Simpler, faster access      |
| 100K - 1M samples          | Either            | Depends on available memory |
| > 1M samples               | StreamingDataset  | Memory efficiency           |
| Multimodal (images/video)  | StreamingDataset  | Large sample sizes          |
| Fast iteration development | Dataset           | Immediate access            |
| Production training        | StreamingDataset  | Scalability                 |
