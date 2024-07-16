"""
This package contains all the callbacks that can be added to the lightning trainer
to visualize information relating to LID estimation.

For example, the callback showing how the LID curve evolvs as we increase time from 0 to 1 is also included here.
"""

from .lidl import MonitorLIDL
from .monitor_lid_curve import MonitorLIDCurve
