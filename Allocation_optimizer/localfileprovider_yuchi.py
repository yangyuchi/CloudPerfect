from __future__ import print_function
import os
from datetime import datetime, timedelta
from dataprovider import DataProvider
from utils import rfc3339_converter, time_as_string
import json


class LocalFileProvider(DataProvider):
    """Data query module

    This class uses the InfluxDB python API to access the database and save the results
    as raw input for the calculator component.

    Parameters
    ----------
    client : InfluxDBClient
    start_time, end_time: datetime

    """
    def __init__(self, client, start_time, end_time):
        super(LocalFileProvider, self).__init__([], [])
        # self.host = 'Host0'
        # self.vms = ['nuberisim-compute-01', 'nuberisim-compute-02', 'nuberisim-compute-03', 'nuberisim-compute-04',
        #             'nuberisim-compute-05', 'nuberisim-compute-06']
        self.host = 'Host1'
        self.vms = ['cfd1', 'cfd2']
        self.client = client
        # by default query the last 24 hours
        self.end_time = end_time
        self.start_time = start_time

    def get_IDs(self):
        with open('hosts_and_vm_names.json', 'w') as f:
            json.dump({self.host: self.vms}, f, indent=2)
        # how to check the IDs from Openstack Compute API

    def get_TimeseriesBandwidthvalues(self):
        """
        The main query function
        :return: Path of the throughput files
        """
        range_as_string = time_as_string(self.start_time) + '_to_' + time_as_string(self.end_time)
        input_folder_path = os.path.join('inputfolder', range_as_string, self.host)
        if not os.path.isdir(input_folder_path):
            os.makedirs(input_folder_path)
        # total should-have number of points
        number_of_points = (self.end_time - self.start_time).total_seconds() / 10
        vms = self.vms
        for vm in vms:
            query_command_tx = "SELECT non_negative_derivative(sum(\"value\"), 1s)*8 FROM \"interface_tx\" " \
                               "WHERE \"type\" = \'if_octets\' AND \"host\" = \'" + vm + "\' AND time >= \'" \
                               + rfc3339_converter(self.start_time) + "\' AND time < \'" \
                               + rfc3339_converter(self.end_time) + "\' GROUP BY time(10s), \"host\" fill(previous)"
            query_command_rx = "SELECT non_negative_derivative(sum(\"value\"), 1s)*8 FROM \"interface_rx\" " \
                               "WHERE \"type\" = \'if_octets\' AND \"host\" = \'" + vm + "\' AND time >= \'" \
                               + rfc3339_converter(self.start_time) + "\' AND time < \'" \
                               + rfc3339_converter(self.end_time) + "\' GROUP BY time(10s), \"host\" fill(previous)"
            data_tx = self.client.query(query_command_tx)
            data_rx = self.client.query(query_command_rx)
            points_tx = [p for p in data_tx.get_points()]
            points_rx = [p for p in data_rx.get_points()]

            if len(points_rx) != number_of_points:
                print("Missing data for VM " + vm + ", remove it from analysis!")
                self.vms.remove(vm)
                continue
            # write values to file
            with open(os.path.join(input_folder_path, vm + '.tx'), 'w') as f:
                for point in points_tx:
                    print("{0} {1}".format(point['time'], point['non_negative_derivative']), file=f)
            with open(os.path.join(input_folder_path, vm + '.rx'), 'w') as f:
                for point in points_rx:
                    print("{0} {1}".format(point['time'], point['non_negative_derivative']), file=f)
        return input_folder_path
