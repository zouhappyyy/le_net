import argparse
import sys
import traceback

from nnunet_mednext.run.run_training import main as _run_main


def main():
    """Wrapper around standard run_training.main with top-level exception capture.

    This keeps the CLI identical to `run_training.py` but prints a clear message
    when training fails due to NaN/Inf or other runtime errors.
    """
    try:
        _run_main()
    except KeyboardInterrupt:
        print("[TRAINING SAFE] Received KeyboardInterrupt, exiting gracefully.")
    except Exception as e:  # noqa: BLE001
        print("[TRAINING SAFE] Unhandled exception during training:", repr(e))
        traceback.print_exc()
        # make sure a non-zero exit code is returned so batch scripts detect failure
        sys.exit(1)


if __name__ == "__main__":
    main()

