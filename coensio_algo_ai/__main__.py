# coensio - entry point for `python -m coensio_algo_ai`
# powered by coesnio.com
import sys

sys.dont_write_bytecode = True

from coensio_algo_ai.cli import main

raise SystemExit(main())
