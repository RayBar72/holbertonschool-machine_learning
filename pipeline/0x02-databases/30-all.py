#!/usr/bin/env python3
"""
30-all
"""
import pymongo


def list_all(mongo_collection):
    """Function that list all documents in a collection

    Args:
        mongo_collection (pymongo): Document in the collection

    Returns:
        list: with the documents
    """
    return mongo_collection.find()
