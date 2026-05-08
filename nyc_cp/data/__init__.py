from nyc_cp.data.loaders import load_actual

__all__ = ["load_actual", "bus", "citibike", "subway", "replica"]


def __getattr__(name):
    # Lazy submodule imports — geopandas is only needed for citibike/subway processing.
    if name in {"bus", "citibike", "subway", "replica"}:
        import importlib

        return importlib.import_module(f"nyc_cp.data.{name}")
    raise AttributeError(name)
