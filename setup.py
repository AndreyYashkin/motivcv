import os
import importlib.util
from itertools import chain

import setuptools

# Вдохновлено:
# https://github.com/NVIDIA/NeMo/blob/main/setup.py


spec = importlib.util.spec_from_file_location("package_info", "motivcv/package_info.py")
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)

__package_name__ = package_info.__package_name__
__version__ = package_info.__version__


def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename), encoding="utf-8") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")

extras_require = {
    # domain packages
    "classification": req_file("classification.txt"),
    "segmentation": req_file("segmentation.txt"),
    "clearml": req_file("clearml.txt"),
}

extras_require["all"] = list(chain(*extras_require.values()))

setuptools.setup(
    name=__package_name__,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    python_requires=">=3.12",
    version=__version__,
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # $ pip install -e ".[all]"
    extras_require=extras_require,
    # Add in any packaged data.
    include_package_data=True,
    exclude=["tests"],
)
