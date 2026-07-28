"""Pure, testable core for the white-box secret-loyalty audit.

The numbered CLI drivers (10_, 11_, 13_) cannot be imported -- Python module
names may not start with a digit -- so every function worth testing lives here
and the drivers stay argparse-only.
"""
