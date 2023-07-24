#!/usr/bin/env python3
"""
update_topics
"""
import pymongo


def update_topics(mongo_collection, name, topics):
    """_summary_

    Args:
        mongo_collection (pymongo): object collection
        name (string): School name to update
        topics (list): topics approached in the school
    """
    mongo_collection.update_many({'name': name}, {'$set': {'topics': topics}})
