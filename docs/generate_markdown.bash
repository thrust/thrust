#! /usr/bin/env bash

###############################################################################
# Copyright (c) 2018 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

cd ${SCRIPT_PATH}/..

rm -rf build_doxygen_xml
rm -rf docs/api
rm -f docs/overview.md
rm -f docs/contributing/code_of_conduct.md
rm -f docs/releases/changelog.md

mkdir docs/api

# We need to copy these files into the `docs/` root because Jekyll doesn't let
# you include content outside of its root.
cp README.md docs/overview.md
cp CODE_OF_CONDUCT.md docs/contributing/code_of_conduct.md
cp CHANGELOG.md docs/releases/changelog.md

doxygen docs/doxygen_config.dox
doxybook2 -d -i build_doxygen_xml -o docs/api -c docs/doxybook_config.json -t docs/doxybook_templates

# Doxygen and Doxybook don't give us a way to disable all the things we'd like,
# so it's important to purge Doxybook Markdown output that we don't need:
# 0) We want our Jekyll build to be as fast as possible and avoid wasting time
#    on stuff we don't need.
# 1) We don't want content that we don't plan to use to either show up on the
#    site index or appear in search results.
rm -rf docs/api/files
rm -rf docs/api/index_files.md
rm -rf docs/api/pages
rm -rf docs/api/index_pages.md
rm -rf docs/api/examples
rm -rf docs/api/index_examples.md
rm -rf docs/api/images
rm -rf docs/api/index_namespaces.md
rm -rf docs/api/index_groups.md
rm -rf docs/api/index_classes.md

