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
    # Load the list of nodes and the list of edges
    csvNodes = "nodes.csv"
    csvEdges = "edges.csv"


    # These intermediary structures will hold the raw csv data for processing
    nodeIDs = dict()
    rawLinks = []
    rawNodes = []

    # These will hold our final node and link objects
    nodes = []
    links = []

    with open(csvNodes) as nodesFile:
        reader = csv.reader(nodesFile, delimiter=",", quotechar="\"")

        nID = 0
        for line in reader:
            params = []

            nodeIDs[line[0]] = nID
            params.append(nID)
            for word in line:
                params.append(word)
            nodes.append(process_node(params))
            nID += 1

    with open(csvEdges) as edgesFile:
        reader = csv.reader(edgesFile,delimiter=",", quotechar="\"")
        for line in reader:
            rawLinks.append(line)

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











