"""The host-side vision tensors, checked against the encoder's own declaration of what it takes.

The previous version of these tests compared the six tensors against ``transformers.vision_utils`` and
passed while five parts of the contract were wrong: a bf16 rotary where the graph declares fp32, an image
split across blocks, two images sharing a block, attention bounds in global instead of block-local
coordinates, and one ``dtype=`` knob covering tensors whose dtypes differ. The reference knows the
arithmetic; only the encoder knows the contract. So these tests read the contract out of the encoder's
own source — its ``forward`` parameter names and the dtypes its warmup declaration builds — rather than
restating it here, where it could drift.

Reading it by parsing the source rather than importing looks indirect, and it is: importing the reference
model pulls in the plugin and vLLM, which is not available where these tests run. The parse is checked
before it is trusted (``test_the_contract_extractor_found_the_declaration``), because an extractor that
silently finds nothing would make every check below vacuous.
"""
import ast
import importlib
import importlib.util
import os
import sys
import types

import pytest
import torch

_search = os.path.dirname(os.path.abspath(__file__))
for _ in range(8):
    _MODEL_DIR = os.path.join(_search, "vllm_neuron", "model", "qwen3_5_moe")
    _REPO = _search
    if os.path.exists(os.path.join(_MODEL_DIR, "vision_inputs.py")):
        break
    _search = os.path.dirname(_search)
else:  # pragma: no cover - only reachable if the test is moved out of the repo
    raise RuntimeError("could not locate vllm_neuron/model/qwen3_5_moe/")

_REF_DIR = os.path.join(_REPO, "vllm_neuron", "model", "qwen3_vl")
_UTILS_DIR = os.path.join(_REF_DIR, "utils")


def _load(path, name, package):
    """Load one module by path, so the plugin (and a device with it) is never imported."""
    stub = types.ModuleType(package)
    stub.__path__ = [os.path.dirname(path)]
    sys.modules.setdefault(package, stub)
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{package}.{name}"] = module
    spec.loader.exec_module(module)
    return module


# vision_inputs imports its helpers through the real package path, so that path has to resolve. Where
# the plugin is genuinely importable (a machine with vLLM), the real package is used and nothing is
# stubbed. Where it is not, a stub is registered under the real names so the helpers can be loaded by
# file.
#
# Both halves of that are the fix for a defect this file caused: it USED to stub unconditionally, and
# pytest imports every test module during collection, so the stub was in place before any test ran and
# shadowed the real package for the whole session. Two checks in another file that import the real
# vision encoder then failed with ModuleNotFoundError -- and passed when run alone, which is the
# signature of collection-time pollution. The stub's ``__path__`` was also wrong by one component
# (``_REPO/model`` for ``vllm_neuron.model``), so even the stubbed path could not have resolved a
# submodule Python had to find on disk.
_pkg = "vllm_neuron.model.qwen3_vl.utils"


def _real_package_importable(module_name: str) -> bool:
    """Whether ``module_name`` imports for real, leaving sys.modules as it was if it does not.

    "For real" is checked rather than assumed. A module already cached under that name satisfies
    ``import_module`` even if a stub put it there, and this function's answer decides whether the shipped
    helpers or the stubs get exercised -- so the resolved module has to have a file on disk. A file
    outside this checkout is also refused: testing the shipped ``vision_inputs.py`` against an older
    INSTALLED copy of its helpers is a green run that means nothing.

    Only ``ImportError`` counts as unavailable. A real package whose import fails for another reason --
    a syntax error, a broken initialiser -- is a regression, and turning it into "use the stubs" would
    hide it behind a suite that still passes.
    """
    before = set(sys.modules)
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        for name in set(sys.modules) - before:
            if name.split(".")[0] == module_name.split(".")[0]:
                del sys.modules[name]
        return False
    origin = getattr(module, "__file__", None)
    if not origin or not os.path.exists(origin):
        return False
    return os.path.realpath(origin).startswith(os.path.realpath(_REPO) + os.sep)


_USING_REAL_PLUGIN = _real_package_importable(f"{_pkg}.vision_block_packing")
if not _USING_REAL_PLUGIN:
    _borrowed = [name for name in ("vllm_neuron", "vllm_neuron.model", "vllm_neuron.model.qwen3_vl",
                                   _pkg) if name not in sys.modules]
    for _part in ("vllm_neuron", "vllm_neuron.model", "vllm_neuron.model.qwen3_vl", _pkg):
        _stub = sys.modules.get(_part) or types.ModuleType(_part)
        # Every component of the dotted name is a directory under the repo root, so the whole name is
        # joined -- not name[1:], which pointed one level above the package.
        _stub.__path__ = getattr(_stub, "__path__", [os.path.join(_REPO, *_part.split("."))])  # lint-port: ok a freshly made ModuleType has no __path__, so the default IS the value being installed
        sys.modules[_part] = _stub
    _load(os.path.join(_UTILS_DIR, "vision_block_packing.py"), "vision_block_packing", _pkg)
    _load(os.path.join(_UTILS_DIR, "vision_preprocessing.py"), "vision_preprocessing", _pkg)

_MINE = "_qwen3_5_moe_vision_under_test"
config = _load(os.path.join(_MODEL_DIR, "config.py"), "config", _MINE)
vision_inputs = _load(os.path.join(_MODEL_DIR, "vision_inputs.py"), "vision_inputs", _MINE)

if not _USING_REAL_PLUGIN:
    # The names are given BACK. Everything that needed them has been executed, and the loaded modules hold
    # their own references, so nothing here depends on the entries surviving.
    #
    # This matters because pytest imports every test module during collection: an entry left behind is
    # visible to the entire session, and a fileless module called `vllm_neuron` sitting in `sys.modules`
    # is exactly what made two checks in a sibling file fail on a module that exists on disk. Leaving it
    # in place also made "restore what I found" a meaningless contract for any test written later, since
    # what it found was already wrong.
    for _part in reversed(_borrowed):
        sys.modules.pop(_part, None)

VISION_CONFIG = config.Qwen3_5MoeVisionConfig()
BLOCK_SIZE = 128
BUCKETS = [128, 256, 512, 1024]


# ---------------------------------------------------------------------------
# The contract, read out of the encoder's source
# ---------------------------------------------------------------------------


def _encoder_forward_parameters():
    """The names of ``Qwen3VLVisionModel.forward``'s tensor parameters, in order."""
    with open(os.path.join(_REF_DIR, "vision_encoder_bf16.py")) as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3VLVisionModel":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return [arg.arg for arg in item.args.args if arg.arg != "self"]
    return []


def _declared_dtypes():
    """``{parameter: dtype}`` from the warmup declaration the runner calls before compiling.

    The warmup tensors are what fix the NEFF's input dtypes, so this is the authority on what the graph
    will accept — not the reference's run-time call site, which could disagree with it.
    """
    with open(os.path.join(_REF_DIR, "model_bf16.py")) as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef)
                and node.name == "build_vision_synthetic_inputs"):
            continue
        for item in ast.walk(node):
            if not isinstance(item, ast.Dict):
                continue
            found = {}
            for key, value in zip(item.keys, item.values):
                if not isinstance(key, ast.Constant) or not isinstance(value, ast.Call):
                    continue
                for keyword in value.keywords:
                    if keyword.arg == "dtype":
                        found[key.value] = ast.unparse(keyword.value).replace("torch.", "")
            if found:
                return found
    return {}


ENCODER_PARAMETERS = _encoder_forward_parameters()
DECLARED_DTYPES = _declared_dtypes()
# The two arguments the caller owns rather than this module: they need the encoder cache.
CALLER_OWNED = {"encoder_cache_buffer", "write_block_ids"}


def test_the_contract_extractor_found_the_declaration():
    """Trust the parse only after it has produced something. A silent empty result would make every
    dtype and key assertion below pass for the wrong reason."""
    assert ENCODER_PARAMETERS, "could not read Qwen3VLVisionModel.forward's parameters"
    assert len(DECLARED_DTYPES) == 7, (
        f"expected 7 declared tensors in build_vision_synthetic_inputs, got {DECLARED_DTYPES}")
    assert set(DECLARED_DTYPES) <= set(ENCODER_PARAMETERS), (
        "the warmup declares a tensor the forward does not take: "
        f"{set(DECLARED_DTYPES) - set(ENCODER_PARAMETERS)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(rows, block_size=BLOCK_SIZE, buckets=None):
    grid = torch.tensor(rows)
    total = int(grid.prod(dim=1).sum())
    pixels = (torch.arange(total * vision_inputs.patch_dim(VISION_CONFIG), dtype=torch.float32)
              .reshape(total, -1) / 1000.0)
    num_blocks = vision_inputs.vision_blocks(grid, block_size, buckets or BUCKETS)
    return vision_inputs.build_vision_inputs(
        pixels, grid, VISION_CONFIG, block_size, num_blocks), grid, num_blocks


# ---------------------------------------------------------------------------
# Keys and dtypes: the two things the reference could not have told me
# ---------------------------------------------------------------------------


def test_keys_are_exactly_the_encoder_parameters_this_module_owns():
    built, _grid, _n = _build([[1, 8, 12]])
    expected = [name for name in ENCODER_PARAMETERS if name not in CALLER_OWNED]
    assert sorted(built) == sorted(expected), (
        "a key that is not a parameter name would be silently dropped by **kwargs, or a missing one "
        "would fall back to a default the encoder does not have")


@pytest.mark.parametrize("name", sorted(set(DECLARED_DTYPES) - CALLER_OWNED))
def test_dtype_matches_the_graphs_declaration(name):
    built, _grid, _n = _build([[1, 8, 12]])
    assert str(built[name].dtype).replace("torch.", "") == DECLARED_DTYPES[name], (
        f"{name} is {built[name].dtype} but the warmup declares {DECLARED_DTYPES[name]}; the NEFF's "
        "input dtype comes from the warmup, so a mismatch is either a load-time rejection or a "
        "silent cast")


def test_rotary_is_fp32_and_not_the_model_dtype():
    """Named separately because this is the defect that shipped: fp32 here is not an accident of the
    helper, it is what the graph takes, and 'the model is bf16' is the reasoning that broke it."""
    built, _grid, _n = _build([[1, 8, 12]])
    assert built["cos"].dtype is torch.float32
    assert built["sin"].dtype is torch.float32


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_shapes_follow_the_bucket_not_the_token_count():
    """A request needing one block still gets the bucket's block count, because the graph was compiled
    for that shape. Sizing from ceil(tokens / block_size) would produce a shape no graph was warmed for.
    """
    built, _grid, num_blocks = _build([[1, 4, 4]], buckets=[512])
    assert num_blocks == 512 // BLOCK_SIZE
    head_dim = VISION_CONFIG.hidden_size // VISION_CONFIG.num_heads
    assert built["pixel_values"].shape == (num_blocks, BLOCK_SIZE,
                                           vision_inputs.patch_dim(VISION_CONFIG))
    assert built["pos_emb_idx"].shape == (4, num_blocks, BLOCK_SIZE)
    assert built["pos_emb_weight"].shape == (4, num_blocks, BLOCK_SIZE)
    assert built["cos"].shape == (num_blocks, BLOCK_SIZE, head_dim)
    assert built["sin"].shape == (num_blocks, BLOCK_SIZE, head_dim)
    assert built["bound_min"].shape == (num_blocks, BLOCK_SIZE, 1)
    assert built["bound_max"].shape == (num_blocks, BLOCK_SIZE, 1)


# ---------------------------------------------------------------------------
# Bounds: block-local, and empty where padded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [
    [[1, 8, 8]],                          # one item, one block
    [[1, 8, 8], [1, 4, 12]],              # two items, one each
    [[1, 8, 8], [1, 8, 8], [1, 4, 8]],    # three items: the case that exposed global coordinates
])
def test_bounds_are_block_local(rows):
    """Every non-empty range must lie inside ``[0, block_size)``.

    Global coordinates satisfy this only for items in the first block, which is exactly why the earlier
    single-image tests could not see the defect: the third item above got the range (128, 160), a window
    outside its own block, and the encoder reads it block-locally.
    """
    built, _grid, num_blocks = _build(rows)
    low = built["bound_min"].reshape(num_blocks, -1)
    high = built["bound_max"].reshape(num_blocks, -1)
    nonempty = high > low
    assert nonempty.any(), "no real patch survived the packing"
    assert int(low[nonempty].min()) >= 0
    assert int(high[nonempty].max()) <= BLOCK_SIZE


def test_padded_rows_see_nothing_and_weigh_nothing():
    """The two padding choices that make a padded row harmless rather than merely deterministic."""
    built, grid, _n = _build([[1, 4, 4]], buckets=[512])
    real = int(grid.prod(dim=1).sum())
    low = built["bound_min"].reshape(-1)
    high = built["bound_max"].reshape(-1)
    # Within the first block, positions at or past the real count are padding.
    padded = slice(real, BLOCK_SIZE)
    assert torch.equal(low[padded], high[padded]), "a padded row was given a non-empty window"
    # And every block after the first is entirely padding.
    assert torch.equal(low[BLOCK_SIZE:], high[BLOCK_SIZE:])
    assert float(built["pos_emb_weight"][:, 1:, :].abs().max()) == 0.0, (
        "a padded bilinear weight is non-zero, so its corner index reaches the embedding table — and "
        "an out-of-range index there is an out-of-bounds gather a compiled graph will not report")


def test_bilinear_weights_sum_to_one_on_real_patches():
    built, grid, _n = _build([[1, 8, 12]])
    real = int(grid.prod(dim=1).sum())
    weights = built["pos_emb_weight"].reshape(4, -1)[:, :real].float().sum(dim=0)
    assert torch.allclose(weights, torch.ones_like(weights), atol=8e-3), (
        f"weights sum to between {weights.min():.4f} and {weights.max():.4f}; bf16 storage allows a "
        "little slack but not a systematic deficit")


# ---------------------------------------------------------------------------
# Legality: the two packing rules the helper enforces and the old code did not
# ---------------------------------------------------------------------------


def test_an_item_larger_than_a_block_is_refused():
    """Splitting one image across blocks severs its attention, because blocks do not attend to each
    other. The old code accepted this and produced an embedding for half an image."""
    with pytest.raises(ValueError, match="exceeds block_size"):
        _build([[1, 12, 16]], buckets=[512])


def test_items_never_share_a_block():
    """One item per block is not an efficiency choice: the encoder writes block i into a single cache
    block and the cache allocates per item, so a shared block has no valid destination."""
    built, grid, num_blocks = _build([[1, 8, 8], [1, 4, 12]])
    tokens = grid.prod(dim=1).tolist()
    # A block holding two items would show two disjoint non-empty ranges starting at 0 and at an
    # offset; one item per block means at most one distinct non-empty range per block.
    low = built["bound_min"].reshape(num_blocks, -1)
    high = built["bound_max"].reshape(num_blocks, -1)
    for block in range(num_blocks):
        ranges = {(int(a), int(b)) for a, b in zip(low[block], high[block]) if b > a}
        assert len(ranges) <= 1, f"block {block} holds {len(ranges)} items: {sorted(ranges)}"
    assert sum(1 for block in range(num_blocks)
               if (high[block] > low[block]).any()) == len(tokens)


def test_pixel_count_must_match_the_grid():
    grid = torch.tensor([[1, 8, 8]])
    pixels = torch.zeros(10, vision_inputs.patch_dim(VISION_CONFIG))
    with pytest.raises(ValueError, match="grid_thw describes"):
        vision_inputs.build_vision_inputs(pixels, grid, VISION_CONFIG, BLOCK_SIZE, 1)


def test_video_is_refused_rather_than_packed_as_one_item():
    """A video's frames pack per frame; treating the item as one T*H*W block forces block_size past the
    whole video and quietly changes what a block means."""
    grid = torch.tensor([[2, 8, 8]])
    total = int(grid.prod(dim=1).sum())
    pixels = torch.zeros(total, vision_inputs.patch_dim(VISION_CONFIG))
    with pytest.raises(NotImplementedError, match="video"):
        vision_inputs.build_vision_inputs(pixels, grid, VISION_CONFIG, 256, 2)


# ---------------------------------------------------------------------------
# cache_write_destinations
# ---------------------------------------------------------------------------


def test_cache_write_destinations_fills_the_tail_with_scratch():
    ids = vision_inputs.cache_write_destinations([[3], [7, 8]], num_blocks=5, scratch_block_id=99)
    assert ids.tolist() == [3, 7, 8, 99, 99]
    assert ids.dtype is torch.int64


def test_cache_write_destinations_refuses_more_cache_blocks_than_the_graph_takes():
    """Dropping the extras would not fail: those tokens would be read back as whatever the cache held."""
    with pytest.raises(ValueError, match="the graph takes"):
        vision_inputs.cache_write_destinations([[1, 2, 3]], num_blocks=2, scratch_block_id=0)


# ---------------------------------------------------------------------------
# The warmup grid formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [128, 256, 1024, 2048])
def test_the_warmup_grid_formula_is_legal_for_a_divisible_block_size(block_size):
    """``[1, merge, block_size // merge]`` fills exactly one block and satisfies the merge grid.

    The warmup computes this rather than searching for a legal side, which is only safe while the block
    size is divisible by ``merge ** 2`` — the factory enforces that. This pins the formula itself: the
    grid must hold exactly ``block_size`` patches, and both its height and width must be multiples of
    the merge size, or the merger consumes partial groups.
    """
    merge = VISION_CONFIG.spatial_merge_size
    assert block_size % (merge ** 2) == 0, "the parametrisation itself must be legal"
    height, width = merge, block_size // merge
    assert height * width == block_size
    assert height % merge == 0 and width % merge == 0
    # And the packer accepts it, which is the claim that matters.
    built, _grid, num_blocks = _build([[1, height, width]], block_size=block_size,
                                      buckets=[block_size])
    assert num_blocks == 1
    assert built["pixel_values"].shape[:2] == (1, block_size)


def test_several_bucket_sized_items_get_one_block_each():
    """The warmup builds ceil(bucket / block_size) items of one block each; the bucket derivation has to
    give them that many blocks. Deriving it from the token total instead would come up short."""
    merge = VISION_CONFIG.spatial_merge_size
    height, width = merge, BLOCK_SIZE // merge
    rows = [[1, height, width]] * 3
    built, _grid, num_blocks = _build(rows, buckets=[BLOCK_SIZE, BLOCK_SIZE * 2, BLOCK_SIZE * 4])
    assert num_blocks >= 3, f"three one-block items need at least three blocks, got {num_blocks}"
    low = built["bound_min"].reshape(num_blocks, -1)
    high = built["bound_max"].reshape(num_blocks, -1)
    occupied = sum(1 for block in range(num_blocks) if (high[block] > low[block]).any())
    assert occupied == 3


def test_cache_write_destinations_refuses_the_scratch_block_as_a_real_destination():
    """A real destination equal to the scratch block puts a padded encoder block and a live one in the
    same place; which lands last is undefined, so the image would be partly overwritten with zeros."""
    with pytest.raises(ValueError, match="scratch block"):
        vision_inputs.cache_write_destinations([[1, 0]], num_blocks=4, scratch_block_id=0)


# ---------------------------------------------------------------------------
# blocks_for_bucket: one conversion for the warmup and the serving path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bucket,block_size,dp,expected", [
    (128, 128, 1, 1),
    (1024, 128, 1, 8),
    (1024, 1024, 1, 1),
    (100, 128, 1, 1),      # a bucket smaller than a block still needs one
    (1024, 128, 4, 8),     # already divisible by dp
    (384, 128, 4, 4),      # 3 blocks rounded up to a multiple of dp
])
def test_blocks_for_bucket(bucket, block_size, dp, expected):
    """The warmup is handed a bucket and must convert it; the serving path selects one and must convert
    it the same way. A second conversion is how the warmed graph ends up with a shape the real path never
    produces -- which surfaces as a load-time rejection at the first image, after the compile is paid."""
    assert vision_inputs.blocks_for_bucket(bucket, block_size, dp) == expected


def test_blocks_for_bucket_refuses_nonsense_divisors():
    for block_size, dp in ((0, 1), (-1, 1), (128, 0)):
        with pytest.raises(ValueError):
            vision_inputs.blocks_for_bucket(1024, block_size, dp)


def test_the_serving_path_and_the_bucket_conversion_agree():
    """vision_blocks selects a bucket and converts it; converting the same bucket directly must give the
    same count, or the two callers disagree about the graph's shape."""
    merge = VISION_CONFIG.spatial_merge_size
    grid = torch.tensor([[1, merge, BLOCK_SIZE // merge]] * 3)
    buckets = [BLOCK_SIZE, BLOCK_SIZE * 2, BLOCK_SIZE * 4, BLOCK_SIZE * 8]
    got = vision_inputs.vision_blocks(grid, BLOCK_SIZE, buckets)
    # The bucket the selection lands on, converted directly.
    chosen = next(b for b in buckets if b >= 3 * BLOCK_SIZE)
    assert got == vision_inputs.blocks_for_bucket(chosen, BLOCK_SIZE, 1)
