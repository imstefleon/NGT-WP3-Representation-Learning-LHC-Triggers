"""
Shared Collide-2V L1 data loading, preprocessing, and split logic.

Extracted from the streaming/preprocessing cells of
`NGT_Linformer_NodesAligned_FullPrecision_Collide2V_L1features.ipynb` so the
logic isn't duplicated across the MHA notebook, `eval_quality_collide2v.py`,
and `scale_test.py`. Behavior (top-pT candidate selection, normalization,
PID one-hot encoding, caching) is kept identical to the notebook.

Usage:
    from ngt_model.datasets.collide2v_data import Collide2VConfig, Collide2VLoader

    cfg = Collide2VConfig(data_source='collide2v_updated', n_constituents=100)
    loader = Collide2VLoader(cfg)
    raw_tensor, y, sample_name = loader.load_classes({'minbias': 2000, 'top': 2000, 'qcd': 2000, 'dy': 2000})
"""

from __future__ import annotations

import fnmatch
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CLASS_NAMES = ['minbias', 'top', 'qcd', 'dy']
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

HF_SAMPLES = {
    'minbias': ['minbias'],
    'top': [
        'tt0123j_5f_ckm_LO_MLM_hadronic',
        'tt0123j_5f_ckm_LO_MLM_leptonic',
        'tt0123j_5f_ckm_LO_MLM_semiLeptonic',
    ],
    'qcd': ['QCD_HT50toInf', 'QCD_HT50tobb'],
    'dy': ['DYJetsToLL_13TeV-madgraphMLM-pythia8'],
}

COLLIDE2V_CLASS_PATH_HINTS = {
    'minbias': ['minbias', 'min_bias', 'minimum_bias'],
    'top': ['top', 'tt_', 'tt-', 'tt0123j', 'ttjets'],
    'qcd': ['qcd'],
    'dy': ['dy', 'drell', 'dyjets'],
}

L1_FEATURE_CANDIDATES = {
    "pt": ["L1PuppiCands_pt", "L1T_PUPPIPart_PT", "L1T_PFCand_PT"],
    "eta": ["L1PuppiCands_eta", "L1T_PUPPIPart_Eta", "L1T_PFCand_Eta"],
    "phi": ["L1PuppiCands_phi", "L1T_PUPPIPart_Phi", "L1T_PFCand_Phi"],
    "dxy": ["L1PuppiCands_dxy", "L1T_PUPPIPart_D0", "L1T_PFCand_D0"],
    "pid": ["L1PuppiCands_pdgId", "L1T_PUPPIPart_PID", "L1T_PFCand_PID"],
}
DEFAULT_COLLIDE2V_PATH = Path('/eos/project/f/foundational-model-dataset/samples/production_final')


def log(msg: str):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


@dataclass
class Collide2VConfig:
    data_source: str = 'collide2v_updated'  # 'collide2v_updated' | 'samples_json' | 'hf'
    n_constituents: int = 100
    feature_mode: str = 'l1_only'
    candidate_selection: str = 'top_pt'
    cache_dir: Path = field(default_factory=lambda: Path.cwd() / 'ngt_mha_nodes_aligned_outputs' / 'cache')
    collide2v_path: Path = field(default_factory=lambda: Path(os.environ.get('COLLIDE2V_PATH', str(DEFAULT_COLLIDE2V_PATH))))
    samples_json_path: Path = field(
        default_factory=lambda: Path(os.environ.get(
            'SAMPLES_JSON_PATH', str(Path.cwd() / 'ctl2_model' / 'data' / '01_raw' / 'samples.json'))))
    root_tree_name: str = field(default_factory=lambda: os.environ.get('ROOT_TREE_NAME', 'Events'))
    dataset_id: str = 'fastmachinelearning/collide-1m'
    raw_read_batch_size: int = 512
    max_schema_mismatches_per_class: int = 10
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))

    def __post_init__(self):
        if self.feature_mode != 'l1_only':
            raise ValueError("feature_mode must be 'l1_only' for NGT runs")
        if self.data_source not in {'collide2v_updated', 'samples_json', 'hf'}:
            raise ValueError(f"data_source must be 'collide2v_updated', 'samples_json', or 'hf'")
        self.cache_dir = Path(self.cache_dir)
        self.collide2v_path = Path(self.collide2v_path)
        self.samples_json_path = Path(self.samples_json_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def feature_set_name(self) -> str:
        return self.feature_mode


class Collide2VLoader:
    """Streams/caches raw (N_CONSTITUENTS, 5) L1 candidate tensors per class."""

    def __init__(self, cfg: Collide2VConfig):
        self.cfg = cfg
        self.feature_candidates = self._build_feature_candidates()
        self._collide2v_parquet_cache: Optional[List[Path]] = None

    def _build_feature_candidates(self) -> Dict[str, List[str]]:
        return {name: list(keys) for name, keys in L1_FEATURE_CANDIDATES.items()}

    def _make_event_tensor(self, arrays: dict) -> Optional[np.ndarray]:
        n_constituents = self.cfg.n_constituents
        min_len = min(len(arrays['pt']), len(arrays['eta']), len(arrays['phi']), len(arrays['dxy']), len(arrays['pid']))
        if min_len == 0:
            return None

        pt_all = np.asarray(arrays['pt'][:min_len], dtype=np.float32)
        valid = np.isfinite(pt_all) & (pt_all > 0)
        if not np.any(valid):
            return None

        # Candidate arrays are not pT-sorted; select the leading candidates by pT
        # before padding/truncation (matches the notebook's top_pt fix).
        valid_idx = np.nonzero(valid)[0]
        order = valid_idx[np.argsort(-pt_all[valid_idx], kind='stable')]
        selected_idx = order[:n_constituents]
        n = len(selected_idx)
        event = np.zeros((n_constituents, 5), dtype=np.float32)
        event[:n, 0] = np.asarray(arrays['pt'], dtype=np.float32)[selected_idx]
        event[:n, 1] = np.asarray(arrays['eta'], dtype=np.float32)[selected_idx]
        event[:n, 2] = np.asarray(arrays['phi'], dtype=np.float32)[selected_idx]
        event[:n, 3] = np.asarray(arrays['dxy'], dtype=np.float32)[selected_idx]
        event[:n, 4] = np.asarray(arrays['pid'], dtype=np.float32)[selected_idx]
        return event

    def _selected_columns(self, available_columns) -> Optional[Dict[str, str]]:
        available = set(available_columns)
        selected = {}
        for feature_name, candidates in self.feature_candidates.items():
            key = next((c for c in candidates if c in available), None)
            if key is None:
                return None
            selected[feature_name] = key
        return selected

    # -- collide2v_updated -------------------------------------------------

    def _list_collide2v_parquet_files(self, class_name: str) -> List[Path]:
        cfg = self.cfg
        if not cfg.collide2v_path.exists():
            raise FileNotFoundError(
                f'collide2v_path does not exist: {cfg.collide2v_path}. '
                'Check EOS access to the production_final folder, or set COLLIDE2V_PATH.')
        if self._collide2v_parquet_cache is None:
            log(f'collide2v_updated: scanning parquet files under {cfg.collide2v_path}')
            self._collide2v_parquet_cache = sorted(cfg.collide2v_path.rglob('*.parquet'))
            log(f'collide2v_updated: found {len(self._collide2v_parquet_cache):,} parquet files')

        hints = [h.lower() for h in COLLIDE2V_CLASS_PATH_HINTS[class_name]]
        matched = []
        for path in self._collide2v_parquet_cache:
            rel = str(path.relative_to(cfg.collide2v_path)).lower()
            if any(hint in rel for hint in hints):
                matched.append(path)
        return matched

    def _stream_from_collide2v_updated(self, class_name: str, target_events: int, cache_prefix: str) -> np.ndarray:
        import pyarrow as pa
        import pyarrow.parquet as pq

        cfg = self.cfg
        cache_path = (cfg.cache_dir /
                      f'{cache_prefix}_{class_name}_{target_events}_{cfg.feature_set_name}_'
                      f'{cfg.candidate_selection}_collide2v_updated_rawtensor.npy')
        if cache_path.exists():
            log(f'{class_name}: loading raw tensor cache {cache_path}')
            return np.load(cache_path, mmap_mode=None)

        files = self._list_collide2v_parquet_files(class_name)
        log('=' * 90)
        log(f'{class_name}: DATA_SOURCE=collide2v_updated, target={target_events:,}, parquet_files={len(files):,}')
        if not files:
            raise RuntimeError(
                f'{class_name}: no updated Collide-2V parquet files matched. '
                f'Check collide2v_path={cfg.collide2v_path} and hints={COLLIDE2V_CLASS_PATH_HINTS[class_name]}')

        events: List[np.ndarray] = []
        selected_seen = None
        files_used = 0
        schema_mismatches = 0

        for path in files:
            if len(events) >= target_events:
                break
            file_start = time.time()
            n_seen = 0
            n_valid = 0
            last_log = time.time()

            try:
                pq_file = pq.ParquetFile(path)
                available_columns = list(pq_file.schema_arrow.names)
                selected = self._selected_columns(available_columns)
                if selected is None:
                    schema_mismatches += 1
                    if schema_mismatches <= cfg.max_schema_mismatches_per_class:
                        log(f'{class_name}: skipping {path.name}; missing L1 columns. '
                            f'available sample={available_columns[:60]}')
                    continue
            except Exception as exc:
                raise RuntimeError(f'{class_name}: could not inspect parquet file {path}. Error: {repr(exc)}') from exc

            selected_seen = selected
            files_used += 1
            columns = sorted(set(selected.values()))
            log(f'{class_name}: reading {path.relative_to(cfg.collide2v_path)}, selected={selected}')

            try:
                for batch in pq_file.iter_batches(columns=columns, batch_size=cfg.raw_read_batch_size):
                    table = pa.Table.from_batches([batch])
                    data = table.to_pydict()
                    n_rows = table.num_rows
                    for row_idx in range(n_rows):
                        arrays = {
                            feature_name: np.asarray(data[column_name][row_idx], dtype=np.float32)
                            for feature_name, column_name in selected.items()
                        }
                        n_seen += 1
                        event = self._make_event_tensor(arrays)
                        if event is not None:
                            events.append(event)
                            n_valid += 1

                        now = time.time()
                        if (n_seen % 1000 == 0) or (now - last_log > 10):
                            elapsed = max(now - file_start, 1e-6)
                            log(f'{class_name}: file_seen={n_seen:,}, file_valid={n_valid:,}, '
                                f'class_loaded={len(events):,}/{target_events:,}, rate={n_seen/elapsed:.1f} ev/s')
                            last_log = now

                        if len(events) >= target_events:
                            break
                    if len(events) >= target_events:
                        break
            except Exception as exc:
                raise RuntimeError(f'{class_name}: could not stream parquet file {path}. Error: {repr(exc)}') from exc

        if len(events) < target_events:
            raise RuntimeError(
                f'{class_name}: loaded only {len(events):,}/{target_events:,} events from updated Collide-2V. '
                f'files_used={files_used}, schema_mismatches={schema_mismatches}, selected_columns={selected_seen}')

        raw = np.stack(events[:target_events], axis=0).astype(np.float32)
        np.save(cache_path, raw)
        log(f'{class_name}: saved raw tensor cache {cache_path}, shape={raw.shape}, files_used={files_used}')
        return raw

    # -- samples_json (ROOT) -----------------------------------------------

    def _load_samples_json_config(self) -> dict:
        import json
        with open(self.cfg.samples_json_path, 'r') as fh:
            data = json.load(fh)
        samples = data.get('samples', data)
        missing = [cls for cls in self.cfg.class_names if cls not in samples]
        if missing:
            raise KeyError(f'samples.json is missing required classes: {missing}. Available: {sorted(samples)}')
        return samples

    def _root_files_for_class(self, class_name: str) -> List[str]:
        samples = self._load_samples_json_config()
        info = samples[class_name]
        files = info.get('files')
        if files is None:
            folder = info.get('folder')
            patterns = info.get('file_pattern', ['*.root'])
            if folder is None:
                raise KeyError(f'{class_name}: samples.json entry must contain either files or folder/file_pattern')
            files = []
            for name in os.listdir(folder):
                if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                    files.append(str(Path(folder) / name))
        files = [str(p) for p in files]
        if not files:
            raise RuntimeError(f'{class_name}: no ROOT files found in samples.json entry')
        return files

    def _stream_from_samples_json(self, class_name: str, target_events: int, cache_prefix: str) -> np.ndarray:
        import uproot
        import awkward as ak

        cfg = self.cfg
        cache_path = (cfg.cache_dir /
                      f'{cache_prefix}_{class_name}_{target_events}_{cfg.feature_set_name}_'
                      f'{cfg.candidate_selection}_samplesjson_rawtensor.npy')
        if cache_path.exists():
            log(f'{class_name}: loading raw tensor cache {cache_path}')
            return np.load(cache_path, mmap_mode=None)

        events: List[np.ndarray] = []
        root_files = self._root_files_for_class(class_name)
        log('=' * 90)
        log(f'{class_name}: DATA_SOURCE=samples_json, target={target_events:,}, root_files={root_files}')

        for root_file in root_files:
            if len(events) >= target_events:
                break
            file_start = time.time()
            n_seen = 0
            n_valid = 0
            last_log = time.time()

            try:
                with uproot.open(root_file) as fh:
                    tree = fh[cfg.root_tree_name]
                    available_columns = list(tree.keys())
                    selected = self._selected_columns(available_columns)
                    if selected is None:
                        log(f'{class_name}: available ROOT branches sample = {available_columns[:80]}')
                        raise RuntimeError(f'{class_name}: missing required L1 particle branches in {root_file}')
                    log(f'{class_name}: selected ROOT branches = {selected}')
            except Exception as exc:
                raise RuntimeError(f'{class_name}: could not inspect ROOT file {root_file}. Error: {repr(exc)}') from exc

            columns = sorted(set(selected.values()))
            try:
                for batch in uproot.iterate(
                    [f'{root_file}:{cfg.root_tree_name}'],
                    expressions=columns,
                    step_size=cfg.raw_read_batch_size,
                    library='ak',
                ):
                    n_rows = len(batch[columns[0]])
                    log(f'{class_name}: decoded ROOT batch rows={n_rows:,} from {root_file}')
                    for row_idx in range(n_rows):
                        arrays = {
                            feature_name: np.asarray(ak.to_numpy(batch[column_name][row_idx]), dtype=np.float32)
                            for feature_name, column_name in selected.items()
                        }
                        n_seen += 1
                        event = self._make_event_tensor(arrays)
                        if event is not None:
                            events.append(event)
                            n_valid += 1

                        now = time.time()
                        if (n_seen % 500 == 0) or (now - last_log > 10):
                            elapsed = max(now - file_start, 1e-6)
                            log(f'{class_name}: seen={n_seen:,}, valid={n_valid:,}, '
                                f'class_loaded={len(events):,}/{target_events:,}, rate={n_seen/elapsed:.1f} ev/s')
                            last_log = now

                        if len(events) >= target_events:
                            break
                    if len(events) >= target_events:
                        break
            except Exception as exc:
                raise RuntimeError(f'{class_name}: could not stream ROOT file {root_file}. Error: {repr(exc)}') from exc

        if len(events) < target_events:
            raise RuntimeError(
                f'{class_name}: loaded only {len(events):,}/{target_events:,} events from samples.json ROOT files.')

        out = np.stack(events[:target_events]).astype(np.float32)
        np.save(cache_path, out)
        log(f'{class_name}: saved raw tensor cache {cache_path}')
        return out

    # -- hf (Hugging Face) ---------------------------------------------------

    def _list_hf_parquet_files(self, folder: str) -> List[str]:
        from huggingface_hub import HfApi

        api = HfApi()
        prefix = f'{folder.strip("/")}/'
        files = api.list_repo_files(repo_id=self.cfg.dataset_id, repo_type='dataset')
        return sorted(f for f in files if f.startswith(prefix) and f.endswith('.parquet'))

    def _stream_from_hf(self, class_name: str, target_events: int, cache_prefix: str) -> np.ndarray:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        cfg = self.cfg
        cache_path = (cfg.cache_dir /
                      f'{cache_prefix}_{class_name}_{target_events}_{cfg.feature_set_name}_'
                      f'{cfg.candidate_selection}_rawtensor.npy')
        if cache_path.exists():
            log(f'{class_name}: loading raw tensor cache {cache_path}')
            return np.load(cache_path, mmap_mode=None)

        fs = HfFileSystem()
        folders = HF_SAMPLES[class_name]
        events: List[np.ndarray] = []
        log('=' * 90)
        log(f'{class_name}: target={target_events:,}, candidate_folders={folders}')

        for folder in folders:
            if len(events) >= target_events:
                break
            folder_start = time.time()
            n_seen = 0
            n_valid = 0
            last_log = time.time()

            try:
                files = self._list_hf_parquet_files(folder)
            except Exception as exc:
                log(f'{class_name}/{folder}: could not list parquet files. Error: {repr(exc)}')
                continue
            if not files:
                log(f'{class_name}/{folder}: no parquet files found')
                continue
            log(f'{class_name}/{folder}: found {len(files):,} parquet files')

            for rel in files:
                if len(events) >= target_events:
                    break
                path = f'hf://datasets/{cfg.dataset_id}/{rel}'
                try:
                    with fs.open(path, 'rb') as fh:
                        pq_file = pq.ParquetFile(fh)
                        available_columns = list(pq_file.schema_arrow.names)
                        selected = self._selected_columns(available_columns)
                        if selected is None:
                            log(f'{class_name}/{folder}: skipping {rel}; missing required particle columns')
                            continue

                        columns = sorted(set(selected.values()))
                        log(f'{class_name}/{folder}: reading {rel}, columns={columns}')
                        for batch in pq_file.iter_batches(columns=columns, batch_size=cfg.raw_read_batch_size):
                            table = pa.Table.from_batches([batch])
                            data = table.to_pydict()
                            for row_idx in range(table.num_rows):
                                arrays = {
                                    feature_name: np.asarray(data[column_name][row_idx], dtype=np.float32)
                                    for feature_name, column_name in selected.items()
                                }
                                n_seen += 1
                                event = self._make_event_tensor(arrays)
                                if event is not None:
                                    events.append(event)
                                    n_valid += 1

                                now = time.time()
                                if (n_seen % 500 == 0) or (now - last_log > 10):
                                    elapsed = max(now - folder_start, 1e-6)
                                    log(f'{class_name}/{folder}: seen={n_seen:,}, valid={n_valid:,}, '
                                        f'class_loaded={len(events):,}/{target_events:,}, rate={n_seen/elapsed:.1f} ev/s')
                                    last_log = now

                                if len(events) >= target_events:
                                    break
                            if len(events) >= target_events:
                                break
                except Exception as exc:
                    log(f'{class_name}/{folder}: failed reading {rel}. Error: {repr(exc)}')
                    continue

            elapsed = time.time() - folder_start
            log(f'{class_name}/{folder}: done seen={n_seen:,}, valid={n_valid:,}, elapsed={elapsed/60:.2f} min')

        if len(events) < target_events:
            raise RuntimeError(f'{class_name}: loaded only {len(events):,}/{target_events:,} events.')

        raw = np.stack(events[:target_events], axis=0).astype(np.float32)
        np.save(cache_path, raw)
        log(f'{class_name}: saved raw tensor cache {cache_path}, shape={raw.shape}')
        return raw

    # -- dispatch / orchestration -------------------------------------------

    def stream_class_raw(self, class_name: str, target_events: int, cache_prefix: str = 'train') -> np.ndarray:
        if self.cfg.data_source == 'collide2v_updated':
            return self._stream_from_collide2v_updated(class_name, target_events, cache_prefix)
        if self.cfg.data_source == 'samples_json':
            return self._stream_from_samples_json(class_name, target_events, cache_prefix)
        if self.cfg.data_source == 'hf':
            return self._stream_from_hf(class_name, target_events, cache_prefix)
        raise ValueError("data_source must be 'collide2v_updated', 'samples_json', or 'hf'")

    def load_classes(
        self, events_per_class: Dict[str, int], cache_prefix: str = 'train'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Streams every class and concatenates into (raw_tensor, y, sample_name, raw_by_class)."""
        raw_by_class = {
            cls: self.stream_class_raw(cls, events_per_class[cls], cache_prefix=cache_prefix)
            for cls in self.cfg.class_names
        }
        raw_tensor = np.concatenate([raw_by_class[cls] for cls in self.cfg.class_names], axis=0).astype(np.float32)
        y = np.concatenate([
            np.full(len(raw_by_class[cls]), LABEL_MAP[cls], dtype=np.int64)
            for cls in self.cfg.class_names
        ])
        sample_name = np.concatenate([
            np.full(len(raw_by_class[cls]), cls, dtype=object)
            for cls in self.cfg.class_names
        ])
        return raw_tensor, y, sample_name, raw_by_class


# ═══════════════════════════════════════════════════════════════════════════
#  preprocessing (identical to eval_quality.py / the notebook)
# ═══════════════════════════════════════════════════════════════════════════

def compute_normalization_constants(data: np.ndarray) -> dict:
    pt = data[:, :, 0]
    eta = data[:, :, 1]
    dxy = data[:, :, 3]
    pt_log = np.log1p(pt)
    return {
        'pt_min': float(pt_log.min()),
        'pt_max': float(pt_log.max()),
        'eta_min': float(eta.min()),
        'eta_max': float(eta.max()),
        'phi_min': -float(np.pi),
        'phi_max': float(np.pi),
        'dxy_min': float(dxy.min()),
        'dxy_max': float(dxy.max()),
    }


def normalize_particles(x: np.ndarray, norm: dict) -> np.ndarray:
    pt_raw = x[..., 0]
    mask_pad = pt_raw == 0

    pt = np.log1p(pt_raw)
    pt = (pt - norm['pt_min']) / (norm['pt_max'] - norm['pt_min'] + 1e-8)
    eta = (x[..., 1] - norm['eta_min']) / (norm['eta_max'] - norm['eta_min'] + 1e-8)
    phi = (x[..., 2] + np.pi) / (2 * np.pi)
    dxy = x[..., 3].copy()
    pid = x[..., 4].copy()

    out = np.stack([pt, eta, phi, dxy, pid], axis=-1).astype(np.float32)
    out[mask_pad, :4] = 0.0
    return out


def build_pid_mapping(data: np.ndarray) -> dict:
    pid_tensor = data[:, :, 4]
    mask_valid = data[:, :, 0] > 0
    valid_pids = pid_tensor[mask_valid].astype(np.int32)
    unique_pids = np.unique(valid_pids)
    return {int(pid): idx for idx, pid in enumerate(unique_pids)}


def map_pid(pid: np.ndarray, pid_mapping: dict) -> np.ndarray:
    mapped = np.full(pid.shape, -1, dtype=np.int32)
    for k, v in pid_mapping.items():
        mapped[pid == k] = v
    return mapped


def preprocess_dataset(
    data: np.ndarray,
    norm: Optional[dict] = None,
    pid_mapping: Optional[dict] = None,
    add_pid_ohe: bool = True,
) -> Tuple[np.ndarray, dict]:
    normed = normalize_particles(data, norm)
    kine = normed[:, :, :4]

    if add_pid_ohe:
        if pid_mapping is None:
            pid_mapping = build_pid_mapping(data)
        pid = normed[:, :, 4].astype(np.int32)
        pid_idx = map_pid(pid, pid_mapping)
        n_pid_classes = len(pid_mapping)
        pid_ohe = np.zeros((*pid_idx.shape, n_pid_classes), dtype=np.float32)
        valid = pid_idx >= 0
        if n_pid_classes > 0:
            pid_ohe[valid] = np.eye(n_pid_classes, dtype=np.float32)[pid_idx[valid]]
        return np.concatenate([kine, pid_ohe], axis=-1).astype(np.float32), pid_mapping

    return normed.astype(np.float32), pid_mapping


def augment_rotate_phi(raw: np.ndarray, max_rotation: float = np.pi, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    aug = raw.copy()
    if rng is None:
        # Matches the notebook's original behavior: draws from the global legacy
        # RNG state (seeded once via np.random.seed(SEED) at notebook start).
        angles = np.random.uniform(-max_rotation, max_rotation, size=(len(raw), 1)).astype(np.float32)
    else:
        angles = rng.uniform(-max_rotation, max_rotation, size=(len(raw), 1)).astype(np.float32)
    mask = aug[:, :, 0] > 0
    aug[:, :, 2] = ((aug[:, :, 2] + angles + np.pi) % (2 * np.pi)) - np.pi
    aug[~mask, 2] = 0.0
    return aug


# ═══════════════════════════════════════════════════════════════════════════
#  splitting
# ═══════════════════════════════════════════════════════════════════════════

def build_split_indices(
    y: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    label_map: Dict[str, int] = LABEL_MAP,
    seed: int = 42,
    balanced_val_events_per_class: Optional[int] = None,
    balanced_eval_events_per_class: Optional[int] = None,
):
    if balanced_val_events_per_class is None and balanced_eval_events_per_class is None:
        rng = np.random.default_rng(seed)
        train_parts, val_parts, eval_parts = [], [], []
        for cls in class_names:
            cls_idx = np.flatnonzero(y == label_map[cls])
            cls_idx = rng.permutation(cls_idx)
            n_temp = int(round(0.30 * len(cls_idx)))
            n_eval = n_temp // 2
            n_val = n_temp - n_eval
            eval_parts.append(cls_idx[:n_eval])
            val_parts.append(cls_idx[n_eval:n_eval + n_val])
            train_parts.append(cls_idx[n_eval + n_val:])
        train_idx = rng.permutation(np.concatenate(train_parts))
        val_idx = rng.permutation(np.concatenate(val_parts))
        eval_idx = rng.permutation(np.concatenate(eval_parts))
        return train_idx, val_idx, eval_idx

    rng = np.random.default_rng(seed)
    train_parts, val_parts, eval_parts = [], [], []
    for cls in class_names:
        cls_idx = np.flatnonzero(y == label_map[cls])
        cls_idx = rng.permutation(cls_idx)
        n_eval = int(balanced_eval_events_per_class or round(0.15 * len(cls_idx)))
        n_val = int(balanced_val_events_per_class or round(0.15 * len(cls_idx)))
        if n_eval + n_val >= len(cls_idx):
            raise ValueError(f'{cls}: requested val+eval={n_val+n_eval:,} but only loaded {len(cls_idx):,} events')
        eval_parts.append(cls_idx[:n_eval])
        val_parts.append(cls_idx[n_eval:n_eval + n_val])
        train_parts.append(cls_idx[n_eval + n_val:])

    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    eval_idx = rng.permutation(np.concatenate(eval_parts))
    return train_idx, val_idx, eval_idx
