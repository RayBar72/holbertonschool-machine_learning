#!/usr/bin/env python3
"""
shcools_by_topic
"""
import pymongo


def schools_by_topic(mongo_collection, topic):
    """List of school having a specific topic

    Args:
        mongo_collection (pymongo): collection object
        topic (string): topic to be searched

    Returns:
        list: of school having an specific topic
    """
    return mongo_collection.find({'topics': {'$in': [topic]}})
