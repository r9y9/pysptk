#!/usr/bin/env bash

set -eu
cd "$(dirname "$0")"

tag=$(git describe --exact-match --tags HEAD)
version=${tag#v}

PYSPTK_BUILD_VERSION="$version" python -m build --sdist
echo "Source distribution is ready at dist/pysptk-$version.tar.gz"
echo "Upload with: python -m twine upload dist/pysptk-$version.tar.gz"
