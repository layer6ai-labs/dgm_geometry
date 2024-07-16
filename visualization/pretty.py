import inspect
from enum import Enum
from typing import List, Optional, Any
import functools
import os

import seaborn as sns
import matplotlib.pyplot as plt


class ColorTheme(Enum):
    """
    Color themes for the plots
    """

    RED_FIRST = "#A02C30"
    RED_SECOND = "#DF8A8A"
    BLUE_FIRST = "#1F78B4"
    BLUE_SECOND = "#A6CBE3"
    GENERATED = "#dfc27d"
    GOLD = "#dfc27d"
    PIRATE_BLACK = "#363838"
    PURPLE = "#5E3C99"
    GREEN = "#5DA05D"
    PIRATE_GOLD = "#BA8003"
    CREAM = "blanchedalmond"
    CHOCOLATE = "chocolate"

    @staticmethod
    def get_colors(count: int) -> List[str]:
        """Returns a list of colors of the given length"""
        if count == 1:
            all_colors = [ColorTheme.RED_FIRST.value]
        elif count == 2:
            all_colors = [ColorTheme.BLUE_FIRST.value, ColorTheme.RED_FIRST.value]
        elif count == 3:
            all_colors = [
                ColorTheme.RED_FIRST.value,
                ColorTheme.BLUE_FIRST.value,
                ColorTheme.GOLD.value,
            ]
        else:
            all_colors = [
                ColorTheme.BLUE_SECOND.value,
                ColorTheme.RED_SECOND.value,
                ColorTheme.PURPLE.value,
                ColorTheme.GREEN.value,
                ColorTheme.GOLD.value,
            ]
        if count <= len(all_colors):
            return all_colors[:count]
        return [f"C{i}" for i in range(count)]


FONT_FAMILY = "serif"

hashlines = ["////", "\\\\\\\\", "|||", "---", "+", "x", "o", "0", ".", "*"]
line_styles = ["-", "--", "-.", ":"]


######################################################
# Generic decorators for the visualization functions #
######################################################


def show_plot(func):
    """
    When this decorator is set, the function calls plt.show() after it returns
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        plt.show()
        # terminal decorator

    return wrapper


class StyleDecorator:
    def __init__(self, font_scale, style, line_style: Optional[str] = None):
        self.font_scale = font_scale
        self.style = style
        self.line_style = line_style

    def __call__(self, func) -> Any:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sns.set(font_scale=self.font_scale, style=self.style)
            if self.line_style:
                plt.rcParams["grid.linestyle"] = self.line_style

            return func(*args, **kwargs)

        return wrapper


def savable(func):
    """
    Takes in a function that presumably plots a figure and adds a parameter
    file_name to it so that it can save the output of the plot in a file.
    """
    # Obtain the signature of the function
    sig = inspect.signature(func)
    # Add the 'file_name' parameter to the signature
    new_params = list(sig.parameters.values()) + [
        inspect.Parameter("file_name", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("transparent_save", inspect.Parameter.KEYWORD_ONLY, default=False),
        inspect.Parameter("dpi", inspect.Parameter.KEYWORD_ONLY, default=300),
    ]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        file_name = kwargs.pop("file_name", None)
        transparent_save = kwargs.pop("transparent_save", False)
        dpi = kwargs.pop("dpi", 300)

        # Execute the function and get the plot
        result = func(*args, **kwargs)

        plt.tight_layout()
        # save it
        if file_name:
            plt.savefig(
                os.path.join(".", "figures", f"{file_name}.png"),
                bbox_inches="tight",
                transparent=transparent_save,
                dpi=dpi,
            )

        return result

    # Update the wrapper's signature
    sig2 = inspect.signature(wrapper)
    new_sig = sig2.replace(parameters=new_params)
    wrapper.__signature__ = new_sig
    return wrapper
