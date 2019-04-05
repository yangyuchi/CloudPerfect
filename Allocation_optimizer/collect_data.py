"""
This file is used to transfer data from remote database to local database.
Please notice the precision issue. If directly import json data to InfluxDB, the precision can only reach millisecond,
For nanosecond precision, must use line protocol and import data from bash script
"""

from __future__ import print_function
from influxdb import InfluxDBClient
import os
import json
from datetime import datetime, timedelta
from requests.exceptions import ChunkedEncodingError


def convert_rfc3339(time):
    return time.isoformat("T") + "Z"


def time_to_string(time):
    return time.strftime("%Y_%m_%d")


# create folders for saving data and bash script
if not os.path.isdir('cloud_data/raw_data'):
    os.makedirs('cloud_data/raw_data')

if not os.path.isdir('cloud_data/import_script'):
    os.makedirs('cloud_data/import_script')

source_client = InfluxDBClient('217.172.12.201', 8086, database='collectd')
target_client = InfluxDBClient('localhost', 8086, database='collectd')

# only query measurements that are available in target database
measurements = [db['name'] for db in target_client.get_list_measurements()]

start_time = datetime.strptime('2019-01-01-00-00', "%Y-%m-%d-%H-%M")
end_time = datetime.strptime('2019-01-01-00-01', "%Y-%m-%d-%H-%M")

with open(os.path.join('cloud_data', 'import_script', 'import_bash_script.sh'), 'w') as f:
    f.write('#!/bin/bash\n')

t1 = start_time
t2 = end_time

while t1 < end_time:
    # query every week, too long period may cause time-out
    t2 = min(t1 + timedelta(weeks=1), end_time)
    for measurement in measurements:
        # get tag names and field names
        tags = source_client.query("SHOW TAG KEYS FROM " + measurement)
        tags = [t['tagKey'] for t in tags.get_points()]
        fields = source_client.query("SHOW FIELD KEYS FROM " + measurement)
        fields = [f['fieldKey'] for f in fields.get_points()]
        query_command = "SELECT * FROM \"" + measurement + "\" WHERE time >= \'" + convert_rfc3339(t1) + \
                        "\' and time < \'" + convert_rfc3339(t2) + "\'"
        json_output = []
        dic = {}  # each dictionary saves a data point
        try:
            data = source_client.query(query_command, epoch='n')
            points = [p for p in data.get_points()]
            # save raw data points as json
            with open(os.path.join('cloud_data', 'raw_data', time_to_string(t1) + '_' + measurement + '.json'), 'w') as f:
                for point in points:
                    dic["measurement"] = measurement
                    tags_dict = {}
                    for tag in tags:
                        tags_dict[tag] = point[tag]
                    fields_dict = {}
                    for field in fields:
                        fields_dict[field] = float(point[field])
                    dic["tags"] = tags_dict
                    dic["fields"] = fields_dict
                    dic["time"] = point['time']
                    json_output.append(dic)
                    dic = {}
                json.dump(json_output, f, indent=2)

            # make the script for data import according to line protocol
            with open(os.path.join('cloud_data', 'import_script', time_to_string(t1) + '_' + measurement + '.txt'), 'w') as f:
                f.write('# DML\n'
                        '# CONTEXT-DATABASE: collectd \n')
                for point in points:
                    import_command = measurement
                    for tag in tags:
                        import_command += (',' + tag + '=' + str(point[tag]))
                    import_command += ' '
                    for field in fields:
                        import_command += (field + '=' + str(point[field]) + ',')
                    import_command = import_command.rstrip(',') + ' '
                    import_command += str(point['time']) + '\n'
                    f.write(import_command)

            # append new command line to the bash script
            with open(os.path.join('cloud_data', 'import_script', 'import_bash_script.sh'), 'a+') as f:
                # !/bin/bash
                f.write('influx -import -path=' + time_to_string(t1) + '_' + measurement + '.txt -precision=ns\n')

        except ChunkedEncodingError:
            print(measurement + ' has too much data, need to reduce period!!!')

    t1 = t2
