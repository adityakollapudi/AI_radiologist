#!/usr/bin/env python
"""pipeline_predict.py: Backward-compatible forwarding pipeline prediction script.

Maintains seamless compatibility with existing scripts and CLI invocations by
forwarding arguments directly to predict.py.
"""
from predict import main

if __name__ == "__main__":
    main()
