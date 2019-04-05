import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from influxdb import DataFrameClient
from datetime import datetime
import os


def convert_rfc3339(time):
    return time.isoformat("T") + "Z"


def time_to_string(time):
    return time.strftime("%Y_%m_%d_%H_%M")


client = DataFrameClient('134.60.152.116', 8086, database='collectd')

if not os.path.isdir('output_decomposition'):
    os.mkdir('output_decomposition')

t1 = datetime.strptime('2019-01-02-00', "%Y-%m-%d-%H")
t2 = datetime.strptime('2019-01-03-00', "%Y-%m-%d-%H")
query_command_tx = "SELECT non_negative_derivative(sum(\"value\"), 1s)*8 FROM \"interface_tx\" " \
                               "WHERE \"type\" = \'if_octets\' AND \"host\" = \'" + 'cfd1' + "\' AND time >= \'" \
                               + convert_rfc3339(t1) + "\' AND time < \'" + convert_rfc3339(t2) + \
                               "\' GROUP BY time(10s), \"host\" fill(null)"
query_command_rx = "SELECT non_negative_derivative(sum(\"value\"), 1s)*8 FROM \"interface_rx\" " \
                   "WHERE \"type\" = \'if_octets\' AND \"host\" = \'" + 'cfd1' + "\' AND time >= \'" \
                   + convert_rfc3339(t1) + "\' AND time < \'" + convert_rfc3339(t2) + \
                   "\' GROUP BY time(10s), \"host\" fill(null)"
data_tx = client.query(query_command_tx)
data_rx = client.query(query_command_rx)
df_tx = data_tx[('interface_tx', (('host', 'cfd1'),))]
df_rx = data_rx[('interface_rx', (('host', 'cfd1'),))]
df = df_tx + df_rx

series = df['non_negative_derivative']
result = seasonal_decompose(series, model='additive', freq=1)
plt.figure(figsize=(20, 10))
result.plot()
plt.tight_layout()
plt.savefig('output_decomposition/decomposition_additive.pdf')
plt.close()

result = seasonal_decompose(series, model='multiplicative', freq=1)
plt.figure(figsize=(20, 10))
result.plot()
plt.tight_layout()
plt.savefig('output_decomposition/decomposition_multiplicative.pdf')
plt.close()

series.rolling(360).mean().plot(figsize=(20, 10), linewidth=5, fontsize=20)
plt.tight_layout()
plt.savefig('output_decomposition/rolling_average.pdf')
plt.close()

series.diff().plot(figsize=(20, 10), linewidth=5, fontsize=20)
plt.tight_layout()
plt.savefig('output_decomposition/1st_order_difference.pdf')
plt.close()
