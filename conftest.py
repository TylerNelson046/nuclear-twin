"""
Root pytest configuration.

Sets NUMBA_DISABLE_JIT=1 before any test modules (and therefore numba) are
imported so that Python's coverage tool can instrument the @njit-decorated
physics kernels (point_kinetics_rhs, xenon_iodine_rhs).  This is the standard
practice for testing Numba-accelerated code:
  https://numba.readthedocs.io/en/stable/reference/envvars.html#numba.NUMBA_DISABLE_JIT

The JIT is only disabled for the test process; production code is unaffected.
"""

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
