from nyc_cp.data.loaders import load_actual

__all__ = ["load_actual", "bus", "subway", "replica"]


def __getattr__(name):
    # Lazy submodule imports — geopandas is only needed for subway processing.
    if name in {"bus", "subway", "replica"}:
        import importlib

        return importlib.import_module(f"nyc_cp.data.{name}")
    raise AttributeError(name)
