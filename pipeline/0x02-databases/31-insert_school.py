#!/usr/bin/env python3
"""
insert_school.py
"""
import pymongo


def insert_school(mongo_collection, **kwargs):
    """Function that inserts a new document

    Args:
        mongo_collection (pymongo): collection object

    Returns:
        _id: returns the identifier
    """
    return mongo_collection.insert_one(kwargs)
