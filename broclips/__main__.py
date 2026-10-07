"""python -m broclips [--port N] [--no-window]  ->  start BroClips (the desktop shortcut runs this with pythonw)."""
import sys

from .server import main

sys.exit(main())
