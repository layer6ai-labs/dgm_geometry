import importlib
import inspect
import pkgutil


def import_package_classes(package):
    """Takes in a package and returns a list of all the subclasses that are defined in that package."""
    classes = []
    # If a package name (string) is provided instead of a module, convert to module
    if isinstance(package, str):
        package = importlib.import_module(package)

    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(name)
        # Iterate through all objects in the module
        for name, obj in inspect.getmembers(module):
            # Check if the object is a class and defined in the module (not imported)
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                classes.append(obj)
        # Recurse into subpackages
        if is_pkg:
            classes.extend(import_package_classes(module))

    return classes
