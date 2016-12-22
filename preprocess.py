#!/usr/bin/env python
# encoding: utf-8
"""
pyedf.py
Created by Ray Slakinski on 2010-09-14.
Copyright (c) 2010 Ray Slakinski. All rights reserved.
"""
import sys
import argparse
import re
import struct
import csv

def read_edf_file(fileobj, fName):
    data = fileobj.read()
    header = {}
    # Parse header information based on the EDF/EDF+ specs
    # http://www.edfplus.info/specs/index.html
    header['version'] = data[0:7].strip()
    header['patient_id'] = data[7:88].strip()
    header['rec_id'] = data[88:168].strip()
    header['startdate'] = data[168:176].strip()
    header['starttime'] = data[176:184].strip()
    header['header_bytes'] = int(data[184:192].strip())
    header['num_items'] = int(data[236:244].strip())
    header['data_duration'] = float(data[244:252].strip())
    header['num_signals'] = int(data[252:256].strip())

    print "Version #: ", header['version']
    print "Patient ID: ", header['patient_id'],
    print "Rec ID: ", header['rec_id']
    print "Start Date: ", header['startdate']
    print "Start Time: ", header['starttime']
    print "Header Bytes: ", header['header_bytes']
    print "Num Items: ", header['num_items']
    print "Data Duration: ", header['data_duration']
    print "Number of Signals: ", header['num_signals']

    off = int(header['header_bytes'])
    dataLength = len(data)
    chLength = header['num_items'] * 256
    print dataLength
    row = []
    for i in range(0,header['num_signals']):
    	row.append(None)
    with open(fName, 'wb') as f:
    	writer = csv.writer(f)
    	for i in range(0, channelLength):
        	for ch in range(0,header['num_signals']):
        		row[ch] = struct.unpack('h', data[(ch * chLength + offset): (ch * chLength + offset+2)])[0]
        	off = off + 2
        	writer.writerows([row])
    


def main():
    # create the parser
    parser = argparse.ArgumentParser(description='Process a given EDF File.')
    parser.add_argument(
        '-file',
        type=argparse.FileType('r'),
        help='EDF File to be processed.',
    )
    args = parser.parse_args()
    fName = args.file.name
    fName = fName[:-4]
    fName = fName + ".csv"
    print fName
    header = read_edf_file(args.file, fName)
    args.file.close()

if __name__ == '__main__':
    main()
