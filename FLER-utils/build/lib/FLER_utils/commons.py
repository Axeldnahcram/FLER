#!/usr/bin/env python
# coding: utf-8
"""
.. module:: albator-utils commons
Common function to retrieve the file content
"""

import os
import logzero
from typing import Dict
import json
import asyncio
import pathlib
from pprint import pprint
# installed
import uvloop
import aioredis
from dotenv import load_dotenv, find_dotenv
import aiofiles
# custom
import sys
import os
sys.path.append(os.path.abspath("~/Documents/Projets/FLER"))
from FLER import constants as cst

LOGGER = logzero.logger

def get_asset_root() -> Dict[str, str]:
    """
    :return: the different assets paths to get sql and csv directories
    :rtype: dictionnary
    """
    dir = pathlib.Path(__file__).parent
    pql_root = str(pathlib.PurePath(dir, 'ext_files/pkl'))
    csv_root = str(pathlib.PurePath(dir, 'ext_files/csv'))
    txt_root = str(pathlib.PurePath(dir, 'ext_files/txt'))
    dict = {}
    dict["pql_root"] = pql_root
    dict["csv_root"] = csv_root
    dict["txt_root"] = txt_root
    return dict

def get_file_content(cfg, name):
    contents = {}
    if cst.CSV_ROOT in cfg:
        root_folder = cfg.get(cst.CSV_ROOT)
        csv_file = f"{root_folder}/{name}.csv"
        if os.path.isfile(csv_file):
            return csv_file
    if cst.PQL_ROOT in cfg:
        root_folder = cfg.get(cst.PQL_ROOT)
        sql_file = f"{root_folder}/{name}.pql"
        if os.path.isfile(pql_file):
            async with aiofiles.open(pql_file) as f:
                contents = await f.read()
    if cst.TXT_ROOT in cfg:
        root_folder = cfg.get(cst.TXT_ROOT)
        txt_file = f"{root_folder}/{name}.txt"
        if os.path.isfile(txt_file):
            return txt_file

        else:
            LOGGER.error("File named %s does not exist", name)
    else:
        LOGGER.error("sql_root is not defined in configuration")
    return contents

if __name__ == "__main__":
    f = get_asset_root()
    g = get_file_content(f, "dffrancais")
    LOGGER.info(g)