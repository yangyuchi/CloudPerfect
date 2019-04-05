from __future__ import print_function
import numpy as np
import os
import pandas as pd
import ConfigParser
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from datetime import timedelta
import pyhsmm
from utils import time_as_string
np.set_printoptions(threshold=np.inf)


class Calculator:
    def __init__(self, configfile, hosts_and_vm_names, input_path, start_time, end_time, report_frequency):
        self.config = ConfigParser.ConfigParser()
        self.config.read(configfile)
        # definition of the bins and bin size
        # bin_size = 10000 for example
        # binsize has to be chosen carefully.
        self.bin_size = self.config.getint('bins', 'bin_size')
        self.min_edge = self.config.getint('bins', 'min_edge')  # 0
        self.max_edge = self.config.getint('bins', 'max_edge')  # 100000 (use 10 Giga, 10 GbE)
        self.bins = np.arange(self.min_edge, self.max_edge + 1, self.bin_size)

        self.start_time = start_time
        self.end_time = end_time
        self.report_frequency = report_frequency

        # get arrays of all hosts and VMs
        self.hosts_and_vm_names = hosts_and_vm_names
        self.hosts = [k for k in hosts_and_vm_names.iterkeys()]
        self.vms = None
        self.input_path = input_path
        self.output_path = None
        self.all_data = None  # all throughput data as DataFrame
        self.df = None  # this DataFrame is used to save sub interval statistics for box plot
        self.summary = None

    def read_raw_data(self):
        """
        Read raw throughput values (tx + rx) of all vms into one DataFrame
        :return: DataFrame of all throughput values for the current host
        """
        df = pd.DataFrame(columns=self.vms)
        for vm in self.vms:
            rx = pd.read_table(self.input_path + '/' + vm + '.rx', header=None, delimiter=" ")
            tx = pd.read_table(self.input_path + '/' + vm + '.tx', header=None, delimiter=" ")
            rx_plus_tx = np.array(rx[1] + tx[1])
            df[vm] = rx_plus_tx
            df.index = rx[0]
        return df

    def hdphmm(self, df):
        """
        Apply Hierarchical Dirichlet Process Hidden Markov Model to analyze the states of throughput,
        use Scalar Gaussian for observation model since our data is 1D,
        use Sticky HDPHMM to prevent frequent fluctuations,
        After getting the hidden state sequence, map them into the rounded mean value of corresponding Gaussian and
        substitute the raw throughput.
        :param df: Original DataFrame of throughput for all vms
        :return: DataFrame with converted throughput
        """
        df[df < 1000000] = 0  # Data rate less than 1Mb convert to 0
        for i, vm in enumerate(df.columns):
            data_vm = df[[vm]][df[vm] >= 1000000]  # For HDPHMM analysis, only consider data rate >= 1Mb
            n_max = 10
            # hyperparameters for prior distribution of Scalar Gaussian
            obs_hypparams = {'mu_0': 0, 'sigmasq_0': 1, 'kappa_0': 0.25, 'nu_0': 3}
            obs_distns = [pyhsmm.distributions.ScalarGaussianNIX(**obs_hypparams) for _ in range(n_max)]
            posterior_model = pyhsmm.models.WeakLimitStickyHDPHMM(kappa=50., alpha=6., gamma=6.,
                                                                  init_state_concentration=1., obs_distns=obs_distns)
            posterior_model.add_data(data_vm.values)
            for _ in range(150):
                posterior_model.resample_model()
            data_vm.loc[:, vm] = posterior_model.stateseqs[0]
            data_vm = data_vm[vm].apply(lambda x: round(obs_distns[int(x)].mu, -3))  # round up to Mb
            df.loc[data_vm.index, vm] = data_vm.values  # substitute raw throughput with Gaussian means
            # plot raw throughput and states
            value_set = df[vm].unique().tolist()
            # get background color sequence including the zero state
            color_sequence = df[vm].apply(lambda x: value_set.index(x)).values
            color_sequence = color_sequence.reshape(1, len(color_sequence))
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(df[vm].values)
            # plot background color
            ax1.pcolorfast(np.arange(df.shape[0]), ax1.get_ylim(), color_sequence, cmap='brg', alpha=0.3)
            ax1.set_xlabel('Data Point ID')
            ax1.set_ylabel('Throughput')
            ax1.set_title('Throughput and states for vm ' + vm)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'throughput_with_states' + vm + '.pdf'))
            plt.close()
        return df

    def do_calculation(self, sample_frequency='10 seconds', apply_hmm=False):
        """
        Main functionality of calculation.
        :param sample_frequency: How often the data points are sampled, by default 10 seconds, the same as in query
        :param apply_hmm: Whether to apply HDPHMM analysis before calculation
        :return:
        """
        assert sample_frequency in ['10 seconds', '1 minute'], "frequency must be 10 seconds or 1 minute!"
        # default parameters for calculation
        sample_step, report_step, num_sub_intervals, td, td_sub = 1, 360, 6, timedelta(hours=1), timedelta(minutes=10)
        if sample_frequency == '1 minute':
            sample_step = 6
        if self.report_frequency == 'daily':
            report_step, num_sub_intervals, td, td_sub = 8640, 24, timedelta(days=1), timedelta(hours=1)
        # get the 2D list of interval strings
        for host in self.hosts:
            print("----------------------------------------------------------")
            print("[  ->  ] Processing server " + host + ", analysis frequency: " + sample_frequency)
            print("----------------------------------------------------------")
            range_as_string = time_as_string(self.start_time) + '_' + time_as_string(self.end_time)
            self.output_path = os.path.join('outputfolder', range_as_string, sample_frequency, host)
            if not os.path.isdir(self.output_path):
                os.makedirs(self.output_path)
            self.vms = self.hosts_and_vm_names[host]
            self.all_data = self.read_raw_data()
            if apply_hmm:
                self.all_data = self.hdphmm(self.all_data)
            t1, t2 = self.start_time, None
            i = 0
            while t1 < self.end_time:
                t2 = t1 + td
                current_interval = time_as_string(t1)
                # create output folders for current interval
                if not os.path.isdir(os.path.join(self.output_path, current_interval)):
                    os.makedirs(os.path.join(self.output_path, current_interval))
                for vm in self.vms:
                    if not os.path.isdir(os.path.join(self.output_path, vm, current_interval)):
                        os.makedirs(os.path.join(self.output_path, vm, current_interval))
                print("----------------------------------------------------------")
                print("[  ->  ] Processing " + self.report_frequency + " data for " + current_interval)
                print("----------------------------------------------------------")
                self.df = pd.DataFrame(columns=pd.MultiIndex.from_product([self.vms, ['placeholder']]))
                raw_data_current_interval = self.all_data.iloc[report_step*i:report_step*(i+1):sample_step, :]
                throughput_values = raw_data_current_interval.T.values
                probabilities = self.calculate_histogram_and_bw_probability(throughput_values, current_interval)
                self.calculate_maximum_data_rate_and_write_to_file(probabilities, current_interval)
                self.calculate_overlay_and_write_to_file(probabilities, host, current_interval)

                # calculate for each sub interval just for boxplot and standard deviation
                j = 0
                while j < num_sub_intervals:
                    sub_interval = time_as_string(t1 + td_sub * j)
                    num_points = raw_data_current_interval.shape[0] / num_sub_intervals
                    raw_data_current_frequency = raw_data_current_interval.iloc[num_points*j:num_points*(j+1), :]
                    throughput_values = raw_data_current_frequency.T.values
                    self.calculate_histogram_and_bw_probability(throughput_values, sub_interval, save=False)
                    j += 1

                self.df = self.df.sort_index(axis=1)
                self.df = self.df.drop(['placeholder'], axis=1, level=1)
                self.df.fillna(0, inplace=True)
                self.df.to_csv(os.path.join(self.output_path, current_interval, 'summary.csv'))
                self.summary = self.df.describe()
                self.summary.to_csv(os.path.join(self.output_path, current_interval, 'statistics.csv'))
                self.plot_boxplot_and_standard_deviation(current_interval)
                t1 = t2
                i += 1

    def calculate_histogram_and_bw_probability(self, throughput_values, interval, save=True):
        """
        Calculate histogram and bandwidth probability distributions for all vms.
        For each report interval, save the results as files and figures,
        For each sub interval, only calculate and save into a DataFrame, which is then used to generate box plots.
        :param throughput_values: Raw throughput values of all vms as a list
        :param interval: current report interval
        :param save: whether to save the result
        :return: probability distribution of different data rates for all vms
        """
        hists, probabilities = [], []
        for n, throughput in enumerate(throughput_values):
            hist = np.histogram(throughput, bins=self.bins)
            v = hist[0] > 0
            reduced_bins = np.array(hist[1][:-1][v])
            reduced_values = np.array(hist[0][v])
            hists.append((reduced_values, reduced_bins))
            probability_list = reduced_values / np.float(sum(reduced_values))
            probabilities.append((probability_list, reduced_bins))
            vm = self.vms[n]
            if save:
                hist_file_name = os.path.join(self.output_path, vm, interval, 'histogram.pdf')
                title = 'Histogram for \n' + vm + ' ' + interval
                self.plot_bar(reduced_values, reduced_bins, hist_file_name, title)
                title = 'Probability Distribution for \n' + vm + ' ' + interval
                prob_file_name = os.path.join(self.output_path, vm, interval, 'bw_prob.pdf')
                self.plot_bar(probability_list, reduced_bins, prob_file_name, title)
            else:
                for i, b in enumerate(reduced_bins):
                    if b not in self.df[vm].columns:
                        self.df[vm, b] = pd.Series(np.zeros(len(self.df.index)), index=self.df.index)
                    self.df.loc[interval, (vm, b)] = probability_list[i]
        if save:
            print("histogram and bandwidth probabilities calculated, writing to files")
            with open(os.path.join(self.output_path, interval, 'hist'), 'w') as f:
                print(hists, file=f)
            with open(os.path.join(self.output_path, interval, 'probabilities'), 'w') as f:
                print(probabilities, file=f)
        return probabilities

    def calculate_maximum_data_rate_and_write_to_file(self, probabilities, interval):
        """
        Maximum data rate with non-zero probability of all vms for current interval
        :param probabilities: Probability list of all vms
        :param interval: current report interval
        :return:
        """
        p, max_data_rates = np.array([]), np.array([])
        for probs, bins in probabilities:
            p = np.concatenate((p, probs))
            max_data_rates = np.concatenate((max_data_rates, bins))
        max_data_rate = [np.max(max_data_rates), p[np.argmax(max_data_rates)]]
        print("maximum data rate calculated, writing to files")
        with open(os.path.join(self.output_path, interval, 'highest'), 'w') as f:
            print (max_data_rate, file=f)

    def calculate_overlay_and_write_to_file(self, probabilities, host_name, interval):
        """
        Overlay calculation, convolution is calculated via Hash Table
        :param probabilities:
        :param host_name:
        :param interval:
        :return:
        """
        first_run = True
        overlay = None
        for probs, bins in probabilities:
            if first_run:
                # Create DataFrame for the first run
                overlay = pd.DataFrame(np.column_stack([bins, probs]), columns=['bins', 'probs'])
                first_run = False
            else:
                hash_table = dict()
                for i in range(len(overlay['bins'])):
                    for j in range(len(bins)):
                        sum_bin = overlay['bins'][i] + bins[j]
                        prod_prob = overlay['probs'][i] * probs[j]
                        hash_table[sum_bin] = hash_table.get(sum_bin, 0) + prod_prob
                new_bins = [k for k in hash_table.iterkeys()]
                new_probs = [v for v in hash_table.itervalues()]
                overlay = pd.DataFrame(np.column_stack([new_bins, new_probs]), columns=['bins', 'probs'])
        print("overlay calculated, writing to files")
        output_csv_name = os.path.join(self.output_path, interval, 'overlay.csv')
        overlay = overlay.sort_values(by=['bins'])
        overlay = overlay.reset_index(drop=True)
        overlay['bins'] = overlay['bins'].astype(np.int)
        overlay.to_csv(output_csv_name)
        output_file_name = os.path.join(self.output_path, interval, 'overlay.pdf')
        title = 'Log Scaled Overlay Probabilities \n for ' + host_name + ' ' + interval
        self.plot_overlay(overlay, output_file_name, title)

    @staticmethod
    def plot_bar(data, bins, save_path, title):
        """
        Plot bar plot for different data rates, when plotting large amount of data, bar plot is much faster than
        histogram in matplotlib.
        :param data: values for each data rate
        :param bins: different data rates
        :param save_path: path to save the figure
        :param title: title of the figure
        :return:
        """
        x = np.arange(len(bins))
        plt.figure(figsize=(6, 6))
        plt.bar(x, data, color='g', edgecolor='b')
        for a, b in zip(x, data):
            if type(b) == np.float64:
                plt.text(a, b, '%.3f' % b, horizontalalignment='center', verticalalignment='bottom')
                plt.ylabel('Probability')
            else:
                plt.text(a, b, str(b), horizontalalignment='center', verticalalignment='bottom')
                plt.ylabel('Frequency')
        plt.title(title)
        plt.xlabel('Data Rate')
        plt.xticks(x, bins, rotation=70)
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_overlay(data, save_path, title):
        """
        Plot overlay_probability for each host and interval, since there are usually lots of overlay data rates,
        xticks only show the start, middle and end data rate. When number of data rate states are too many, use
        normal plot other than bar plot for speed.
        :param data: Calculated overlay probability distribution as DataFrame
        :param save_path: Path for saving the plot
        :param title: title of the plot
        :return:
        """
        x = np.array(data['bins'])
        overlay = np.array(data['probs'])
        x_pos = np.arange(len(x))
        x_ticks = [0, len(x) // 2, len(x) - 1]
        x_tick_labels = [x[tick] for tick in x_ticks]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        if len(x) <= 100:
            ax.bar(x_pos, overlay, color='g')
        else:
            ax.plot(x_pos, overlay, color='k')
            ax.fill_between(x_pos, 0, overlay, color='k')
        ax.set_yscale('log')
        ax.set_xlabel('Data Rate')
        plt.ticklabel_format(style='plain', axis='x', scilimits=(0, 0))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylabel('Overlay Probability')
        ax.set_ylim = [min(overlay), 1]
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_boxplot_and_standard_deviation(self, interval):
        """
        Function for plotting box plots and standard deviations (error bars).
        For the sake of clarity, only data rates that have enough number of sub intervals with non-zero probability
        are plotted.
        :param interval: current report interval
        :return:
        """
        print("plotting box plots and standard deviations")
        for vm in self.df.columns.levels[0]:
            plt.figure(figsize=(6, 6))
            data = self.df[vm]
            # count_non_zeros = (data != 0).sum(axis=0)  # count number of sub_intervals where probability is not zero
            # index_columns = count_non_zeros[count_non_zeros >= (data.shape[0] - 1)].index
            # data = data.loc[:, index_columns]  # only select data rates with enough valid sub_intervals
            statistics = self.summary[vm]
            # statistics = statistics.loc[:, index_columns]
            boxplot = data.boxplot(rot=70, return_type='dict')
            plt.xlabel('Data Rates')
            plt.ylabel('Probability')
            plt.title('Boxplot of BW probability \n for VM: ' + vm + ' ' + interval)
            plt.grid(linestyle='-.', alpha=0.1)
            # customize text shown on box plots
            for line in boxplot['medians']:
                x, y = line.get_xydata()[1]
                plt.text(x, y, '%.2f' % y, fontsize=4, horizontalalignment='left', verticalalignment='center')
            for line in boxplot['boxes']:
                x, y = line.get_xydata()[1]
                plt.text(x, y, '%.2f' % y, fontsize=4, horizontalalignment='left', verticalalignment='center')
                x, y = line.get_xydata()[2]
                plt.text(x, y, '%.2f' % y, fontsize=4, horizontalalignment='left', verticalalignment='center')
            for line in boxplot['whiskers']:
                x, y = line.get_xydata()[1]
                plt.text(x, y, '%.2f' % y, fontsize=4, horizontalalignment='center', verticalalignment='bottom')
            save_path = os.path.join(self.output_path, vm, interval, 'boxplot.pdf')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            plt.figure(figsize=(6, 6))
            plt.xlabel('Data Rates')
            plt.ylabel('Mean and Standard Deviation')
            plt.title('Error bar of BW probability \n for VM: ' + vm + ' ' + interval)
            plt.grid(linestyle='-.', alpha=0.1)
            mean = statistics.loc['mean'].tolist()
            std = statistics.loc['std'].tolist()
            mins = statistics.loc['min'].tolist()
            maxes = statistics.loc['max'].tolist()
            stability, colors = [], []
            for i in range(len(mean)):
                if mins[i] < mean[i] - 3 * std[i] or maxes[i] > mean[i] + 3 * std[i]:
                    stability.append('unstable')
                    colors.append('r')
                else:
                    stability.append('stable')
                    colors.append('k')

            plt.errorbar(np.arange(len(statistics.columns)) + 1, mean, std, fmt='.k', ecolor=colors)
            # customize text shown on error bars
            for x, m, s in zip(np.arange(len(statistics.columns)) + 1, mean, std):
                plt.text(x, m+s, '{:.2f}'.format(m) + u'\u00b1' + '{:.2f}'.format(s), fontsize=4,
                         horizontalalignment='center', verticalalignment='bottom')
            save_path = os.path.join(self.output_path, vm, interval, 'standard_deviation.pdf')
            plt.xticks(np.arange(len(statistics.columns)) + 1, statistics.columns.values.tolist(), rotation=70)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
