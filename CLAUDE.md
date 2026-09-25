# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Repast4Py is a distributed agent-based modeling (ABM) framework: a Python package over two C++ CPython extensions, parallelized with MPI. `dev_notes.md` holds additional maintainer detail (debugging segfaults under MPI, asciidoc manual, PyPI upload).

## Two build modes

Everything in this repo exists in two variants, selected at **build** time:

| | default (native MPI) | single-rank ("mock") |
|---|---|---|
| trigger | none | `R4PY_NO_MPI=1` |
| C headers | system `<mpi.h>`, `<mpi4py/mpi4py.h>` | bundled stubs in `src/repast4py/single_rank_include/` |
| Python `MPI` | `mpi4py.MPI` | `repast4py._mpi_stub.MPI` (pure Python, size 1, no-op collectives) |
| needs | `mpicxx`, mpi4py | C++ compiler only |

`setup.py` writes the generated marker [src/repast4py/_mpi_config.py](src/repast4py/_mpi_config.py) at build time; [_mpi.py](src/repast4py/_mpi.py) reads it to pick the Python MPI, so the Python side always matches what `_space` was compiled against (env var is only a fallback for an unbuilt source tree). mpi4py is a *dynamic* build requirement supplied by the in-tree PEP 517 backend [build_support/r4py_backend.py](build_support/r4py_backend.py) — that is why `pyproject.toml` sets `build-backend = "r4py_backend"` and omits mpi4py from `requires`.

A single-rank build launched under `mpirun`/`srun` detects launcher env vars and exits with an error ([_mpi_stub.py](src/repast4py/_mpi_stub.py) `_check_launch`), since the mock would otherwise silently produce N independent runs.

## Build

```bash
# default, in place (both CC and CXX are needed on newer setuptools)
CC=mpicxx CXX=mpicxx python setup.py build_ext --inplace
# debug symbols
CC=mpicxx CXX=mpicxx CFLAGS="-O0 -g" CXXFLAGS="-O0 -g" python setup.py build_ext --inplace
# single-rank, no MPI toolchain required
R4PY_NO_MPI=1 python setup.py build_ext --inplace

CC=mpicxx CXX=mpicxx pip install -e .       # installs
R4PY_NO_MPI=1 pip install -e .         # works under normal build isolation
```

A build always rewrites `_mpi_config.py`, so switching modes requires a rebuild — the in-tree `_*.so` and the marker must agree.

**Switching modes needs a forced rebuild.** `build_ext` only compares timestamps, and changing the mode changes no source file, so a second `build_ext --inplace` in the other mode recompiles nothing and relinks the *previous* mode's objects. The marker flips but the `.so` does not, and the result is silent: the stub's `PyMPIComm_Get` ignores the communicator it is handed, so a stub-built `_space` paired with real mpi4py still imports and still constructs a `SharedGrid` at size 1 — it would only misbehave under `mpirun`, as independent non-communicating ranks, and the launch guard cannot catch it because the Python side is genuinely mpi4py. Delete the objects when switching:

```bash
rm -rf build/temp.* build/lib.* src/repast4py/_*.so src/repast4py/_mpi_config.py
```

## Test

**Most of the suite needs MPI, and `discover` does not find it.** `dev_notes.md` ("Compiling and Testing") is the authoritative list of the four categories and their rank counts; it is not optional reading before touching the distributed code. Roughly 88 of ~155 tests only run under `mpirun`, and they are the ones that exercise cross-rank movement, ghosting, buffers, and the shared network — a green `discover` run has tested none of that.

Two reasons the MPI tests are easy to miss: `discover` uses the default `test*.py` pattern, so the `*_tests.py` modules are invisible to it and must be named explicitly; and each of those modules skips itself at size 1 rather than failing, so running one without `mpirun` reports success having run nothing.

```bash
# single-process only: the 6 test_*.py modules
python -m unittest discover tests
python -m unittest tests.test_space.GridTests.test_move         # one test

# the MPI suites (default build only; rank counts are required, not suggestions)
mpirun -n 9  python -m unittest tests.shared_obj_tests          # 2D spaces
mpirun -n 9  python -m unittest tests.shared_vl_tests
mpirun -n 9  python -m unittest tests.ctopo_tests
mpirun -n 18 python -m unittest tests.shared_obj_tests.SharedGridTests.test_buffer_data_3d          # 3D spaces
mpirun -n 18 python -m unittest tests.shared_obj_tests.SharedGridTests.test_buffer_data_3d_periodic
mpirun -n 4  python -m unittest tests.logging_tests
mpirun -n 4  python -m unittest tests.shared_network_tests
```

The rank counts match how each suite partitions its global bounds, so a suite run at the wrong `-n` is not a weaker test but a meaningless one.

`./run_tests.sh` walks all of the above interactively (prompting between suites); `tests/test.sh` runs them non-interactively under `coverage` and sums exit codes — use that one for a full check. Add `--oversubscribe` when there are fewer cores than ranks. `RDMAV_FORK_SAFE=1` is set in CI and tox.

`tests/util_tests.py` is in neither script nor CI, and `discover` cannot see it — it runs only if named explicitly.

A parallel test module must guard itself so a single-rank run skips it rather than failing:

```python
def setUpModule():
    if MPI.COMM_WORLD.Get_size() == 1:
        raise unittest.SkipTest('requires more than one rank (run with mpirun)')
```

Both modes across Python 3.9–3.13 via tox (`py{39,310,311,312,313}-{mpi,mock}`); tox builds non-editably per env because a shared install cannot represent both modes:

```bash
tox            # everything
tox -f mpi     # native MPI: discover + all the mpirun suites
tox -f mock    # single-rank: discover only
tox -e py312-mpi
```

`tox.ini` encodes the same split: the `mpi:`-prefixed `commands` are the `mpirun` lines, so only the `mpi` envs run the distributed suites — the `mock` envs stop at `discover`. **A green `tox -f mock` therefore proves the single-rank build works, not that the distributed code does**; the same is true of CI's `test-single-rank` job, which sets up no MPI at all. Cross-rank changes must be checked under `tox -f mpi`, `tests/test.sh`, or the `mpirun` lines directly.

Because the rank counts and suite list live in four places that must agree — `tox.ini`, [.github/workflows/tests.yml](.github/workflows/tests.yml), `run_tests.sh`, and `tests/test.sh` — adding or renaming an MPI suite means updating all four (`dev_notes.md` lists them as the fifth).

C++ gtest tests: copy `tests/Makefile` to a `Release`/`Debug` dir at top level, edit the hardcoded include/lib paths, `make tests`, then `mpirun -n 9 ./unit_tests --gtest_filter=CartesianTopology.*`.

Lint: `flake8` (config in `setup.cfg`, ignores E501/W503).

## Architecture

**Native layer.** `_core` ([coremodule.cpp](src/repast4py/coremodule.cpp)) defines the `Agent` base type and an agent iterator; `_space` ([spacemodule.cpp](src/repast4py/spacemodule.cpp), ~3k lines) defines `DiscretePoint`, `ContinuousPoint`, `Grid`, `ContinuousSpace`, `SharedGrid`, `SharedContinuousSpace`, `CartesianTopology`, borders. `_space` reaches `_core`'s types through a `PyCapsule` C API (`repast4py._core._C_API`, see [coremodule.h](src/repast4py/coremodule.h) / `import_core()`), so **`_core` must import successfully before `_space`**. `_space` is the only extension that touches MPI, via `PyMPIComm_Get` and the C `MPI_*` calls the stub header mirrors.

**Python layer wraps native classes by subclassing them.** `space.SharedGrid(_SharedGrid, SharedProjection)` is the pattern — C++ holds the spatial data structures and buffer/ghost exchange; Python adds the synchronization protocol and Pythonic API. `BorderType` / `OccupancyType` in [space.py](src/repast4py/space.py) are integer enums duplicated in the C++ grid/space init code; changing a value means changing both.

**`SharedContext` is the hub.** [context.py](src/repast4py/context.py) holds the agent population for one rank and a set of *projections* (spatial, network) that impose structure on them. `synchronize(restore_agent)` is a collective that, in order: pre-synchs ghosts → gathers out-of-bounds agents from bounded projections and `alltoall`s them → recreates moved agents on the destination rank → notifies every projection that agents moved → synchs ghost state → post-move fixups. Agent movement across ranks is therefore driven entirely by projections reporting out-of-bounds agents; the context never knows about geometry.

**Projections implement two Protocols** defined in [core.py](src/repast4py/core.py): `SharedProjection` (the `_pre_synch_ghosts` / `_synch_ghosts` / `_agent_moving_rank` / `_agents_moved_rank` / `_post_agents_moved_rank` callbacks the context calls during synchronize) and `BoundedProjection` (`_get_oob` / `_clear_oob` / `_move_oob_agent`, for projections an agent can leave). Spatial projections are both; networks are only `SharedProjection`. Adding a projection type means implementing these hooks — they are the cross-rank contract.

**Agent identity and ghosting.** An agent's `uid` is `(id, type, rank-of-origin)`. `AgentManager` (internal) tracks local agents, *ghosts* (read-only copies of remote agents, ref-counted by how many projections reference them), and *ghosted* agents (locals copied out, mapped to the ranks holding them). Agents cross ranks by `save()`/`restore` — user models supply a `restore_agent` callable that reconstructs an agent from its `save()` tuple. Ghost state is refreshed each synchronize by `alltoall` of `save()` payloads.

**Other modules.** [schedule.py](src/repast4py/schedule.py) — `SharedScheduleRunner` keeps ranks in lockstep by taking the global min next-event tick each iteration; events have `PriorityType` (FIRST/RANDOM/BY_PRIORITY/LAST) ordering within a tick; accessed via module-level `init_schedule_runner()` / `runner()`. [network.py](src/repast4py/network.py) — networkx-backed shared graphs, plus `read_network`/`write_network` with random or metis partitioning. [value_layer.py](src/repast4py/value_layer.py) — PyTorch-tensor-backed N-d value grids with their own buffer synchronization; `ReadWriteValueLayer` double-buffers. [logging.py](src/repast4py/logging.py) — `ReducingDataLogger` (cross-rank reduction of dataclass fields) and `TabularLogger`. [checkpoint.py](src/repast4py/checkpoint.py) — dill-pickled snapshot of random state, schedule, agents, spaces, networks. [random.py](src/repast4py/random.py) — the shared `default_rng`; use it rather than a private generator so checkpointing and `shuffle` stay reproducible. [parameters.py](src/repast4py/parameters.py) — yaml params file + CLI override parsing.

**Model idiom** (see [examples/zombies/zombies.py](examples/zombies/zombies.py)): a `Model` class owning a `SharedContext`, projections, a schedule runner, and loggers; agent classes subclass `core.Agent` with `save()`; a module-level `restore_agent`; `run(params)` entry point driven by `create_args_parser()` + `init_params()` against a yaml file.

## Conventions

- Import MPI as `from repast4py import MPI` — never `from mpi4py import MPI`. Direct mpi4py imports break the single-rank build. This applies to library code, tests, and examples.
- Any new MPI C symbol used in `_space` must be added to `src/repast4py/single_rank_include/mpi.h`, or the single-rank build stops compiling.
- New `.h` files and anything under `single_rank_include/` need a `MANIFEST.in` entry to reach the sdist.
- Google-style docstrings with PEP 484 hints (types in hints are not repeated in the docstring). Sphinx docs live in `docs/`; a new module needs `sphinx-apidoc -e -o source ../src/repast4py` from `docs/`, then `make html`.
- `develop` is the integration branch; `master` is the release branch.
