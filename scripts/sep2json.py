#!/usr/bin/env python
""" 
Script to convert SEP LDA Edges csv file to a JSON data file for use with D3.
"""

import sys
import csv
import json

def process_node(nodeParams):
    """
    Converts a row of node data to a named dictionary for easy JSON output.
    """
    node = dict()
    node["id"] = nodeParams[0]
    node["name"] = nodeParams[1]
    node["group"] = int(nodeParams[2])
    return node 

def process_link(linkParams):
    """
    Converts a row of link data to a named dictionary for easy JSON output.
    """
    link = dict()
    link["source"] = int(linkParams[0])
    link["target"] = int(linkParams[1])
    link["value"] = 1
    return link

if __name__ == "__main__":
    csvFile = sys.argv[-1]


    # These intermediary structures will hold the raw csv data for processing
    nodeSet = set()
    nodeIDs = dict()
    rawLinks = []

    # These will hold our final node and link objects
    nodes = []
    links = []

    with open(csvFile) as f:
        reader = csv.reader(f,delimiter=",", quotechar="\"")
        for line in reader:
            for word in line:
                # Add nodes to nodeSet to remove duplicates 
                nodeSet.add(word)
            rawLinks.append(line)

    # Create our actual nodes
    nID = 0
    for node in nodeSet:
        params = []

        # Add nodes and nIDs to nodeIDs for easy association
        nodeIDs[node] = nID

        # Temporary static groupID
        groupID = 1
        params.append(nID)
        params.append(node)
        params.append(groupID)
        nodes.append(process_node(params))

        nID += 1

    # Replace explicit names in links with node ids and process
    for edge in rawLinks:
        params = []
        for node in edge:
            node = nodeIDs[node]
            params.append(node)
        links.append(process_link(params))

    data = dict()
    data["nodes"] = nodes
    data["links"] = links

    result = json.dumps(data)
    outFile = open(raw_input("Save to what filename?: "), 'w')
    outFile.write(result)











