#!/usr/bin/env python3
"""
30-all
"""
import pymongo


def list_all(mongo_collection):
    return mongo_collection.find()
