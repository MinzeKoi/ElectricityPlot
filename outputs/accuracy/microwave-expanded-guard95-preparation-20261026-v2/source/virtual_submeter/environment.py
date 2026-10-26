"""Record the machine actually measured; never invent a second developer machine."""

import ctypes
from datetime import datetime, timezone
import os
import platform
import shutil
import sys


def environment_report() -> dict:
    memory = None
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            memory = {"total_bytes": status.total, "available_bytes": status.available}
    elif hasattr(os, "sysconf"):
        try:
            memory = {"total_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")}
        except (ValueError, OSError):
            pass
    disk = shutil.disk_usage(".")
    return {
        "schema_version": 1, "project": "ElectricityPlot", "measured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "system": platform.system(), "architecture": platform.machine(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(), "memory": memory, "workspace_disk_free_bytes": disk.free,
        "m1_runtime_dependencies": [], "m1_execution": "CPU / Python standard library",
        "training_environment": "not_frozen", "second_developer_machine": "not_measured",
        "notice": "GPU training compatibility has not been validated. M2 dependencies remain pending the second machine measurement.",
    }
