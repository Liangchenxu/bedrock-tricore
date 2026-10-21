import sys
import traceback

sys.modules["pynvml"] = None

import run_v10_eligibility as m

try:
    m.main()
except Exception:
    with open("_v10_err.log", "w", encoding="utf-8") as handle:
        handle.write(traceback.format_exc())
    raise
