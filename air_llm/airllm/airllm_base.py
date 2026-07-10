
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer

from .cpu_layer_cache import CPULayerCache
from .profiler import LayeredProfiler

from .utils import clean_memory, load_layer, \
    find_or_create_local_splitted_path

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False


class _LiveStats:
    """Throttled terminal stats for the active streaming group."""

    def __init__(self, enabled=False, interval=0.25, stream=None):
        self.enabled = enabled
        self.interval = max(0.0, float(interval))
        self.stream = stream or sys.stderr
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.started_at = time.monotonic()
        self.last_emit_at = 0.0
        self.last_line_length = 0

    def start(self):
        self.started_at = time.monotonic()
        self.last_emit_at = 0.0

    def emit(self, *, forward, batch_size, sequence_length, group_id, group_count,
              layers, current_layer, phase, prefetch, force=False):
        if not self.enabled:
            return

        now = time.monotonic()
        if not force and now - self.last_emit_at < self.interval:
            return
        if force and not self.interactive and phase == "compute" \
                and now - self.last_emit_at < self.interval:
            return

        elapsed = now - self.started_at
        sequence = sequence_length if sequence_length is not None else "?"
        line = (
            f"[AirLLM] forward={forward} batch={batch_size} seq={sequence} "
            f"phase={phase} group={group_id + 1}/{group_count} "
            f"layers={layers} current={current_layer} "
            f"prefetch={prefetch} elapsed={elapsed:.1f}s"
        )
        if self.interactive:
            padding = max(0, self.last_line_length - len(line))
            print("\r" + line + (" " * padding), end="", file=self.stream, flush=True)
            self.last_line_length = len(line)
        else:
            print(line, file=self.stream, flush=True)
        self.last_emit_at = now

    def finish(self):
        if self.enabled and self.interactive:
            print(file=self.stream, flush=True)


class AirLLMBaseModel:
    """
    Memory-frugal wrapper around a Hugging Face ``*ForCausalLM`` model.

    The checkpoint is split into per-layer shards on disk. The real transformers model is
    instantiated on the ``meta`` device (no memory used) and owns the full forward / generation
    logic. AirLLM only attaches forward hooks to each big module (embeddings, every decoder
    layer, the final norm and the lm_head) to stream groups of module weights disk -> GPU right
    before they run and free the whole group after the last module, prefetching the next group
    on a worker thread.

    Because transformers drives the forward pass, AirLLM no longer needs to track per-architecture
    attention/rotary/cache details: new model architectures work as soon as transformers supports
    them.
    """

    # Subclasses override this to point at non-standard module names.
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                                 'layer_prefix': 'model.layers',
                                 'norm': 'model.norm',
                                 'lm_head': 'lm_head'}

    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 layers_per_gpu_group=1, prefetch_groups=1, show_live_stats=False,
                 live_stats_interval=0.25, cuda_copy_stream=True,
                 cpu_layer_cache_gib=0.0):
        """
        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            runtime dtype; defaults to the model's own config.torch_dtype (usually bfloat16 for
            modern models). float16 has too narrow a range for very deep models and overflows to
            inf/NaN, which silently corrupts the output, so we don't force it.
        max_seq_len : int, optional
            max seq length, by default 512
        layer_shards_saving_path : str, optional
            optional path to save the splitted shards, by default next to the model cache
        profiling_mode : bool, optional
            whether to profile the model loading time, default False
        compression: str, optional
            '4bit' or '8bit' to enable block-wise quantization of the on-disk shards
        hf_token: str, optional
            huggingface api token
        prefetching: bool, optional
            overlap the next group's disk load with the current group's compute
        delete_original: bool, optional
            delete the original downloaded checkpoint after splitting to save disk space
        layers_per_gpu_group: int, optional
            number of consecutive decoder layers to keep resident on the GPU at once. The
            embedding, final norm, and lm_head are always kept as separate edge groups.
        prefetch_groups: int, optional
            number of upcoming GPU groups to load into CPU memory ahead of execution.
        show_live_stats: bool, optional
            print throttled group/layer/prefetch progress to stderr during generation.
        live_stats_interval: float, optional
            minimum seconds between non-critical live-stat updates.
        cuda_copy_stream: bool, optional
            stage the immediate next group on a dedicated CUDA stream while the
            current group computes. Disabled automatically on CPU or without prefetching.
        cpu_layer_cache_gib: float, optional
            bounded CPU RAM budget for retaining layer shards across forward passes.
            Cached tensors are pinned when prefetching is enabled so the CUDA copy
            stream can reuse them without another disk read or staging copy.
        """

        if not isinstance(layers_per_gpu_group, int) or isinstance(layers_per_gpu_group, bool) \
                or layers_per_gpu_group < 1:
            raise ValueError("layers_per_gpu_group must be a positive integer")
        if not isinstance(prefetch_groups, int) or isinstance(prefetch_groups, bool) \
                or prefetch_groups < 1:
            raise ValueError("prefetch_groups must be a positive integer")
        if live_stats_interval < 0:
            raise ValueError("live_stats_interval must be non-negative")
        if isinstance(cpu_layer_cache_gib, bool) or not isinstance(
                cpu_layer_cache_gib, (int, float)) or cpu_layer_cache_gib < 0:
            raise ValueError("cpu_layer_cache_gib must be a non-negative number")

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self.hf_quantizer = None

        if compression is not None and not bitsandbytes_installed:
            raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. '
                              'To use compression, please install bitsandbytes: `pip install bitsandbytes`')

        self.compression = compression
        self.hf_token = hf_token
        self.layers_per_gpu_group = layers_per_gpu_group
        self.prefetch_groups = prefetch_groups

        self.set_layer_names_dict()

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            compression=compression,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original)

        self.running_device = device
        self.device = torch.device(self.running_device)

        # Prefer transformers' native implementation; only trust the model's bundled remote code when
        # transformers doesn't recognize the architecture. Vendored remote code is frequently pinned
        # to an old transformers and breaks against the current cache/generation APIs (e.g.
        # DeepSeek-V2's modeling_deepseek.py calls the long-removed DynamicCache.seen_tokens).
        token_kwargs = {'token': hf_token} if hf_token is not None else {}
        try:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=False, **token_kwargs)
            self.trust_remote_code = False
        except Exception:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=True, **token_kwargs)
            self.trust_remote_code = True

        # Default to the model's native dtype (bf16 for most modern models). Forcing fp16 overflows
        # on deep models (e.g. Qwen3-235B's 94 layers) and produces garbage; bf16's wider range
        # avoids it. Users can still override via dtype=.
        if dtype is None:
            cfg_dtype = getattr(self.config, "torch_dtype", None)
            if isinstance(cfg_dtype, str):
                cfg_dtype = getattr(torch, cfg_dtype, None)
            dtype = cfg_dtype if isinstance(cfg_dtype, torch.dtype) else torch.float16
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        self.generation_config = self.get_generation_config()
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        # prefetch executor / state
        self.prefetching = prefetching
        if self.compression is not None and self.prefetching:
            print("prefetching is not supported together with compression for now; disabling prefetching.")
            self.prefetching = False
        self._executor = ThreadPoolExecutor(max_workers=1) if self.prefetching else None
        self._prefetch_futures = {}
        self.cuda_copy_stream = bool(
            cuda_copy_stream and self.prefetching and self.device.type == "cuda")
        self._copy_stream = torch.cuda.Stream(device=self.device) if self.cuda_copy_stream else None
        self._gpu_prefetch_executor = (
            ThreadPoolExecutor(max_workers=1) if self.cuda_copy_stream else None)
        self._gpu_prefetch_futures = {}
        self._group_cpu_sources = {}
        self._group_copy_events = {}
        self.cpu_layer_cache = CPULayerCache(
            max_gib=float(cpu_layer_cache_gib),
            pin_memory=self.prefetching and self.device.type == "cuda",
        )
        self._resident_group_id = None
        self._forward_count = 0
        self._live_batch_size = "?"
        self._live_sequence_length = "?"
        self._live_stats = _LiveStats(show_live_stats, live_stats_interval)
        self._reset_runtime_stats()

        self.init_model()

        # compute layer count from the instantiated model
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        layers_count = len(model_attr)

        self.layer_names = [self.layer_names_dict['embed']] + \
                           [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.set_layers_from_layer_names()
        self._install_streaming_hooks()

    # ---- customization hooks for subclasses -------------------------------------------------

    def get_generation_config(self):
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    # ---- model construction -----------------------------------------------------------------

    def init_model(self):
        # Build the real model on meta (no memory). include_buffers=False so non-persistent
        # buffers such as rotary inv_freq are actually computed (they aren't in the checkpoint).
        self.model = None
        try:
            with init_empty_weights(include_buffers=False):
                self.model = AutoModelForCausalLM.from_config(
                    self.config, attn_implementation="sdpa", trust_remote_code=self.trust_remote_code)
        except (ValueError, TypeError) as e:
            print(f"attn_implementation='sdpa' not available ({e}), falling back to eager attention")
            self.model = None
        if self.model is None:
            # Some (often remote-code) architectures don't support sdpa and also default to it, so we
            # must request eager explicitly; otherwise transformers re-selects sdpa and errors again.
            with init_empty_weights(include_buffers=False):
                self.model = AutoModelForCausalLM.from_config(
                    self.config, attn_implementation="eager", trust_remote_code=self.trust_remote_code)

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

        self.model.eval()
        self.model.tie_weights()
        self.model.generation_config = self.generation_config

        # Move all (already-materialized) buffers to the running device, preserving their dtype.
        # This includes rotary inv_freq, which transformers computes once at the model level and
        # passes down to every decoder layer.
        for buffer_name, buffer in self.model.named_buffers():
            if buffer is not None and buffer.device.type != 'meta':
                set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer)

        # Force the model to report the running (cuda) device even though its parameters live on
        # meta between layer executions, so transformers' generation utilities place inputs/cache
        # tensors on the right device.
        self._patch_device_property()

    def _patch_device_property(self):
        running_device = torch.device(self.running_device)
        running_dtype = self.running_dtype
        base_cls = type(self.model)

        class _AirLLMRuntimeModel(base_cls):
            @property
            def device(self):
                return running_device

            @property
            def dtype(self):
                return running_dtype

        self.model.__class__ = _AirLLMRuntimeModel

    def set_layers_from_layer_names(self):
        self.layers = []

        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    # ---- weight streaming -------------------------------------------------------------------

    def load_layer_to_cpu(self, layer_name):
        def read_layer():
            started = time.time()
            output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
            elapsed_time = time.time() - started

            if self.profiling_mode:
                state_dict, compression_time = output
                disk_loading_time = elapsed_time - compression_time
                self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)
                self.profiler.add_profiling_time('compression_time', compression_time)
                return state_dict
            return output

        return self.cpu_layer_cache.get_or_load(layer_name, read_layer)

    def move_layer_to_device(self, state_dict):
        moved = []
        for param_name in self._param_names_from_state_dict(state_dict):
            if self.hf_quantizer is not None and self._needs_quantization(param_name):
                # On-the-fly-quantizing schemes (e.g. bitsandbytes) reconstruct the param from the
                # weight plus companion quant-state tensors carried in state_dict.
                self.hf_quantizer.create_quantized_param(self.model, state_dict[param_name], param_name,
                                                         self.running_device, state_dict)
            else:
                # Normal load. Pre-quantized weights (fp8) and their block scales must be placed
                # verbatim: casting an fp8 weight to fp16 silently drops the quantization and the
                # accompanying weight_scale_inv, producing garbage. Only ordinary high-precision
                # tensors get cast to the runtime dtype.
                value = state_dict[param_name]
                if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or param_name.endswith("_scale_inv"):
                    if value.device.type == "cpu" and value.is_pinned() and self.device.type == "cuda":
                        value = value.to(self.running_device, non_blocking=True)
                    set_module_tensor_to_device(self.model, param_name, self.running_device, value=value)
                else:
                    if value.device.type == "cpu" and value.is_pinned() and self.device.type == "cuda":
                        value = value.to(self.running_device, non_blocking=True)
                    set_module_tensor_to_device(self.model, param_name, self.running_device,
                                                value=value, dtype=self.running_dtype)
            moved.append(param_name)
        return moved

    def _needs_quantization(self, param_name):
        q = self.hf_quantizer
        # transformers renamed check_quantized_param -> param_needs_quantization.
        if hasattr(q, "param_needs_quantization"):
            return q.param_needs_quantization(self.model, param_name)
        return q.check_quantized_param(self.model, param_value=None, param_name=param_name, state_dict={})

    def _param_names_from_state_dict(self, state_dict):
        names = []
        for param_name in state_dict.keys():
            # bitsandbytes stores a weight plus companion quant-state tensors named
            # "<weight>.4bit.*" / "<weight>.8bit.*"; those are reconstructed together via
            # create_quantized_param, so collapse them down to the base weight name. Everything
            # else (including fp8 weight + weight_scale_inv pairs) is kept as distinct params.
            if '.4bit.' in param_name or '.8bit.' in param_name:
                base = param_name.split('.4bit.')[0].split('.8bit.')[0]
                if base not in names:
                    names.append(base)
            elif param_name not in names:
                names.append(param_name)
        return names

    def _install_streaming_hooks(self):
        # Modules execute in this order during a forward: embed -> layers -> norm -> lm_head.
        n = len(self.layer_names)

        # Detect tied input/output embeddings. When tied, lm_head shares the embedding weight, so
        # there is no separate lm_head shard. We keep the embedding resident on the GPU (it is the
        # only copy and such models are small) and re-tie lm_head to it, then stream only the
        # decoder layers and the final norm.
        self.tie_word_embeddings = bool(getattr(self.config, "tie_word_embeddings", False))
        self._resident_embedding_moved = []

        if self.tie_word_embeddings:
            embed_state = self.load_layer_to_cpu(self.layer_names[0])
            self._resident_embedding_moved = self.move_layer_to_device(embed_state)
            self.model.tie_weights()
        self._streaming_groups = self._build_streaming_groups(
            n, self.tie_word_embeddings, self.layers_per_gpu_group)
        self._streamed_indices = [idx for group in self._streaming_groups for idx in group]

        self._streamed_set = set(self._streamed_indices)
        self._group_by_idx = {
            idx: group_id
            for group_id, group in enumerate(self._streaming_groups)
            for idx in group
        }
        self._group_last_idx = {
            group_id: group[-1]
            for group_id, group in enumerate(self._streaming_groups)
        }

        for idx in self._streamed_indices:
            module = self.layers[idx]
            module._airllm_idx = idx
            module.register_forward_pre_hook(self._pre_hook)
            module.register_forward_hook(self._post_hook)

    @staticmethod
    def _build_streaming_groups(layer_count, tie_word_embeddings, layers_per_gpu_group):
        """Build decoder groups while keeping model edges in separate groups."""
        if layer_count < 3:
            raise ValueError("AirLLM requires embedding, decoder, and output layers")
        if not isinstance(layers_per_gpu_group, int) or isinstance(layers_per_gpu_group, bool) \
                or layers_per_gpu_group < 1:
            raise ValueError("layers_per_gpu_group must be a positive integer")

        groups = []
        decoder_indices = list(range(1, layer_count - 2))

        if not tie_word_embeddings:
            groups.append([0])

        for start in range(0, len(decoder_indices), layers_per_gpu_group):
            groups.append(decoder_indices[start:start + layers_per_gpu_group])

        # Keep the final norm separate from decoder groups so it does not extend the
        # residency window unexpectedly. A tied lm_head has no separate shard.
        groups.append([layer_count - 2])
        if not tie_word_embeddings:
            groups.append([layer_count - 1])
        return groups

    def _load_group_to_cpu(self, group_id):
        return [
            self.load_layer_to_cpu(self.layer_names[idx])
            for idx in self._streaming_groups[group_id]
        ]

    def _schedule_group_prefetch(self, group_id):
        if not self.prefetching or group_id is None:
            return
        last_group_id = min(
            len(self._streaming_groups),
            group_id + 1 + self.prefetch_groups,
        )
        for next_group_id in range(group_id + 1, last_group_id):
            if next_group_id not in self._prefetch_futures:
                self._prefetch_futures[next_group_id] = self._executor.submit(
                    self._load_group_to_cpu, next_group_id)

    def _take_group_from_prefetch(self, group_id):
        if self.prefetching and group_id in self._prefetch_futures:
            state_dicts = self._prefetch_futures.pop(group_id).result()
            return state_dicts
        return self._load_group_to_cpu(group_id)

    def _group_supports_cuda_prefetch(self, state_dicts):
        if self.hf_quantizer is None:
            return True
        return not any(
            self._needs_quantization(param_name)
            for state_dict in state_dicts
            for param_name in self._param_names_from_state_dict(state_dict)
        )

    def _materialize_group(self, group_id, state_dicts):
        expected_layers = self._streaming_groups[group_id]
        if len(state_dicts) != len(expected_layers):
            raise RuntimeError(
                f"prefetched group {group_id} contained {len(state_dicts)} layers; "
                f"expected {len(expected_layers)}"
            )

        loaded_modules = []
        try:
            for idx, state_dict in zip(expected_layers, state_dicts):
                module = self.layers[idx]
                module._airllm_moved = self.move_layer_to_device(state_dict)
                loaded_modules.append(module)
        except Exception:
            for module in loaded_modules:
                if self.hf_quantizer is not None:
                    for param_name in getattr(module, '_airllm_moved', []):
                        set_module_tensor_to_device(self.model, param_name, 'meta')
                else:
                    module.to('meta')
                module._airllm_moved = []
            raise
        return loaded_modules

    def _prepare_group_on_cuda_stream(self, group_id, cpu_future):
        state_dicts = cpu_future.result()
        if not self._group_supports_cuda_prefetch(state_dicts):
            return {"async": False, "state_dicts": state_dicts}

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(self.device), torch.cuda.stream(self._copy_stream):
            start_event.record(self._copy_stream)
            self._materialize_group(group_id, state_dicts)
            end_event.record(self._copy_stream)
        return {
            "async": True,
            "state_dicts": state_dicts,
            "start_event": start_event,
            "end_event": end_event,
        }

    def _schedule_cuda_group_prefetch(self, group_id):
        if not self.cuda_copy_stream:
            return
        next_group_id = group_id + 1
        if next_group_id >= len(self._streaming_groups) \
                or next_group_id in self._gpu_prefetch_futures:
            return
        cpu_future = self._prefetch_futures.pop(next_group_id, None)
        if cpu_future is None:
            return
        self._gpu_prefetch_futures[next_group_id] = self._gpu_prefetch_executor.submit(
            self._prepare_group_on_cuda_stream, next_group_id, cpu_future)

    def _record_group_on_stream(self, group_id, stream):
        for idx in self._streaming_groups[group_id]:
            module = self.layers[idx]
            for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
                if tensor is not None and tensor.device.type == "cuda":
                    tensor.record_stream(stream)

    def _activate_cuda_prefetched_group(self, group_id):
        future = self._gpu_prefetch_futures.pop(group_id, None)
        if future is None:
            return False, None

        wait_started = time.perf_counter()
        prepared = future.result()
        self._runtime_stats["group_cpu_wait_seconds"] += time.perf_counter() - wait_started
        if not prepared["async"]:
            return False, prepared["state_dicts"]

        end_event = prepared["end_event"]
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(end_event)
        copy_wait_started = time.perf_counter()
        end_event.synchronize()
        self._runtime_stats["group_copy_wait_seconds"] += time.perf_counter() - copy_wait_started
        self._runtime_stats["group_gpu_load_seconds"] += (
            prepared["start_event"].elapsed_time(end_event) / 1000.0)
        self._record_group_on_stream(group_id, current_stream)
        self._group_cpu_sources[group_id] = prepared["state_dicts"]
        self._group_copy_events[group_id] = end_event
        self._runtime_stats["groups_loaded"] += 1
        self._runtime_stats["cuda_prefetched_groups"] += 1
        self._resident_group_id = group_id
        return True, None

    def _prefetch_status(self):
        if not self._prefetch_futures and not self._gpu_prefetch_futures:
            return "none"
        cpu_status = [
            f"{group_id + 1}:{'ready' if future.done() else 'loading'}"
            for group_id, future in sorted(self._prefetch_futures.items())
        ]
        gpu_status = [
            f"{group_id + 1}:{'gpu-ready' if future.done() else 'copying'}"
            for group_id, future in sorted(self._gpu_prefetch_futures.items())
        ]
        return ",".join(cpu_status + gpu_status)

    def _emit_live_stats(self, group_id, idx, phase, args=None, force=False):
        group_layers = self._streaming_groups[group_id]
        current_position = group_layers.index(idx) + 1
        self._live_stats.emit(
            forward=self._forward_count,
            batch_size=self._live_batch_size,
            sequence_length=self._live_sequence_length,
            group_id=group_id,
            group_count=len(self._streaming_groups),
            layers=f"{current_position}/{len(group_layers)} "
                   f"({','.join(str(layer_idx) for layer_idx in group_layers)})",
            current_layer=self.layer_names[idx],
            phase=phase,
            prefetch=self._prefetch_status(),
            force=force,
        )

    def _unload_group(self, group_id):
        self._release_group_modules(group_id)
        self._group_cpu_sources.pop(group_id, None)
        self._group_copy_events.pop(group_id, None)
        if self._resident_group_id == group_id:
            self._resident_group_id = None

    def _release_group_modules(self, group_id):
        for idx in self._streaming_groups[group_id]:
            module = self.layers[idx]
            if self.hf_quantizer is not None:
                for param_name in getattr(module, '_airllm_moved', []):
                    set_module_tensor_to_device(self.model, param_name, 'meta')
            else:
                module.to('meta')
            module._airllm_moved = []

    def _load_group_to_device(self, group_id):
        first_idx = self._streaming_groups[group_id][0]
        self._emit_live_stats(group_id, first_idx, "loading", force=True)
        activated, state_dicts = self._activate_cuda_prefetched_group(group_id)
        if activated:
            self._schedule_group_prefetch(group_id)
            self._schedule_cuda_group_prefetch(group_id)
            self._emit_live_stats(group_id, first_idx, "ready", force=True)
            return

        if state_dicts is None:
            cpu_wait_started = time.perf_counter()
            state_dicts = self._take_group_from_prefetch(group_id)
            self._runtime_stats["group_cpu_wait_seconds"] += time.perf_counter() - cpu_wait_started
        gpu_load_started = time.perf_counter()
        try:
            self._materialize_group(group_id, state_dicts)
        except Exception:
            clean_memory()
            raise

        self._runtime_stats["group_gpu_load_seconds"] += time.perf_counter() - gpu_load_started
        self._runtime_stats["groups_loaded"] += 1
        self._resident_group_id = group_id
        self._schedule_group_prefetch(group_id)
        self._schedule_cuda_group_prefetch(group_id)
        self._emit_live_stats(group_id, first_idx, "ready", force=True)

    def _pre_hook(self, module, args):
        idx = module._airllm_idx
        group_id = self._group_by_idx[idx]

        if group_id == 0 and idx == self._streaming_groups[group_id][0]:
            self._forward_count += 1
            try:
                shape = args[0].shape
                self._live_batch_size = int(shape[0])
                self._live_sequence_length = int(shape[1]) if len(shape) > 1 else "?"
            except (AttributeError, IndexError, TypeError, ValueError):
                self._live_batch_size = "?"
                self._live_sequence_length = "?"

        if self._resident_group_id != group_id:
            if self._resident_group_id is not None:
                self._unload_group(self._resident_group_id)
            self._load_group_to_device(group_id)
        if idx == self._streaming_groups[group_id][0]:
            self._group_compute_started[group_id] = time.perf_counter()
        self._emit_live_stats(group_id, idx, "compute", args=args)

    def _post_hook(self, module, args, output):
        idx = module._airllm_idx
        group_id = self._group_by_idx[idx]
        if self._group_last_idx[group_id] == idx:
            compute_started = self._group_compute_started.pop(group_id, None)
            if compute_started is not None:
                self._runtime_stats["group_compute_seconds"] += time.perf_counter() - compute_started
            self._emit_live_stats(group_id, idx, "release", force=True)
            self._unload_group(group_id)
        return output

    # ---- delegation to the underlying transformers model ------------------------------------

    def _reset_runtime_stats(self):
        self._runtime_stats = {
            "groups_loaded": 0,
            "group_cpu_wait_seconds": 0.0,
            "group_gpu_load_seconds": 0.0,
            "group_copy_wait_seconds": 0.0,
            "group_compute_seconds": 0.0,
            "cuda_prefetched_groups": 0,
        }
        self._group_compute_started = {}

    def _cancel_prefetch_futures(self):
        cpu_futures = tuple(self._prefetch_futures.values())
        self._prefetch_futures.clear()
        for future in cpu_futures:
            if future.cancel():
                continue
            try:
                future.result()
            except BaseException:
                pass
        gpu_futures = tuple(self._gpu_prefetch_futures.items())
        self._gpu_prefetch_futures.clear()
        for group_id, future in gpu_futures:
            if future.cancel():
                continue
            try:
                prepared = future.result()
            except BaseException:
                continue
            if prepared.get("async"):
                prepared["end_event"].synchronize()
                self._release_group_modules(group_id)

    def close(self):
        """Release resident weights and stop the CPU prefetch worker."""
        self._cancel_prefetch_futures()
        if self._resident_group_id is not None:
            self._unload_group(self._resident_group_id)
        if getattr(self, "tie_word_embeddings", False):
            if self.hf_quantizer is not None:
                for param_name in self._resident_embedding_moved:
                    set_module_tensor_to_device(self.model, param_name, 'meta')
            else:
                self.layers[0].to('meta')
            self._resident_embedding_moved = []
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if self._gpu_prefetch_executor is not None:
            self._gpu_prefetch_executor.shutdown(wait=True, cancel_futures=True)
            self._gpu_prefetch_executor = None
        self.cpu_layer_cache.close()
        clean_memory()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def get_runtime_stats(self):
        """Return timing counters for the most recent generation call."""
        cache_stats = self.cpu_layer_cache.stats()
        return {
            **self._runtime_stats,
            "forward_passes": self._forward_count,
            "layers_per_gpu_group": self.layers_per_gpu_group,
            "prefetch_groups": self.prefetch_groups,
            "cuda_copy_stream": self.cuda_copy_stream,
            "cpu_cache_hits": cache_stats.hits,
            "cpu_cache_misses": cache_stats.misses,
            "cpu_cache_evictions": cache_stats.evictions,
            "cpu_cache_bytes": cache_stats.bytes,
            "cpu_cache_max_bytes": cache_stats.max_bytes,
            "cpu_cache_entries": cache_stats.entries,
        }

    def generate(self, *args, **kwargs):
        self._forward_count = 0
        self._live_batch_size = "?"
        self._live_sequence_length = "?"
        self._reset_runtime_stats()
        self._live_stats.start()
        try:
            return self.model.generate(*args, **kwargs)
        except BaseException:
            self._cancel_prefetch_futures()
            if self._resident_group_id is not None:
                self._unload_group(self._resident_group_id)
            clean_memory()
            raise
        finally:
            self._live_stats.finish()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
