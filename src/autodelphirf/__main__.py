"""``python -m autodelphirf``: the same command line as ``autodelphirf``.

Running through the interpreter guarantees the code that runs is the code
that interpreter's pip installed, whatever ``autodelphirf`` is first on PATH.
"""
from .cli import main

raise SystemExit(main())
