from calculator_yuchi import Calculator
from config import ConfigInitializer
from localfileprovider_yuchi import LocalFileProvider
from influxdb import InfluxDBClient
import json
from datetime import datetime
from utils import set_query_period


SET_PERIOD_MANUALLY = False

# -----------------------------------------------------------------------------------
#    createConfig(file, min_edge, max_edge, bin_size, first_bin, overbookingfactor):
# -----------------------------------------------------------------------------------

configFile = "configfile.ini"
cfgInit = ConfigInitializer()
cfgInit.createConfig(configFile, 0, 10000000000, 1000, 1000, 8000000000)

report_frequency = 'daily'
assert report_frequency in ['hourly', 'daily'], "report frequency must be hourly or daily!"
start_time = datetime.strptime('2019-01-02-00', "%Y-%m-%d-%H")
end_time = datetime.strptime('2019-01-09-00', "%Y-%m-%d-%H")
# optionally set query period from user input
if SET_PERIOD_MANUALLY:
    start_time, end_time = set_query_period()


client = InfluxDBClient('', 8086, database='collectd')
lfp = LocalFileProvider(client, start_time, end_time)

print "----------------------------------------------------------"
print "- [Stage 1] Collecting data                              -"
print "----------------------------------------------------------"

input_path = lfp.get_TimeseriesBandwidthvalues()
lfp.get_IDs()

print "----------------------------------------------------------"
print "- [Stage 2] Calculating results                          -"
print "----------------------------------------------------------"

# the json file is currently of no use because we have to specify hosts and vms manually
hosts_and_vm_names_file = "hosts_and_vm_names.json"
with open(hosts_and_vm_names_file, 'r') as f:
    hosts_and_vm_names = json.load(f)
calculator = Calculator(configFile, hosts_and_vm_names, input_path, start_time, end_time, report_frequency)
calculator.do_calculation(apply_hmm=True)
calculator.do_calculation(sample_frequency='1 minute', apply_hmm=True)
