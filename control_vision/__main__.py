import os
import sys


def _ensure_std_streams() -> None:
    """Give the windowed (console-less) Windows build somewhere to write.

    PyInstaller sets sys.stdout and sys.stderr to None when there is no
    console, and libraries that print or log to them would then fail.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _configure_frozen_runtime() -> None:
    """Keep libraries from treating the frozen executable as a Python interpreter.

    Ultralytics checks optional packages when it loads a model and, by default,
    runs ``sys.executable -m pip install`` for any it cannot find. In the frozen
    build sys.executable is ControlVision.exe, so that would start a second copy
    of the app and wait for it. The bundle is complete as built; disable the
    installer and the network probe that goes with it.
    """
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _ensure_std_streams()
    frozen = bool(getattr(sys, "frozen", False))
    if frozen:
        if argv[:1] in (["-m"], ["-c"]):
            # Something ran sys.executable as Python; never open the GUI for it.
            print("ControlVision.exe is not a Python interpreter.", file=sys.stderr)
            return 2
        _configure_frozen_runtime()
    if "--smoke-test" in argv:
        from control_vision.smoke_test import main as smoke_main
        return smoke_main(argv)
    from control_vision.app.main_window import run
    run()
    return 0


if __name__ == "__main__":
    # A frozen build re-runs this entry point for any multiprocessing child.
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
