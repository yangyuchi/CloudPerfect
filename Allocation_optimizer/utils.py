from datetime import datetime


def rfc3339_converter(time):
    """
    Convert time to RFC3339 format
    :param time: time as datetime format
    :return: RFC3339 format time
    """
    return time.isoformat("T") + "Z"


def time_as_string(time):
    """
    Convert time to a string connected by underscore
    :param time: time as datetime format
    :return: a string
    """
    return time.strftime("%Y_%m_%d_%H_%M")


def set_query_period():
    """
    Set the query period manually
    :return: Start and end time of the period
    """
    while True:
        start_time_input = raw_input("Input the start date and hour in YYYY-MM-DD-HH format:\n")
        end_time_input = raw_input("Input the end date and hour in YYYY-MM-DD-HH format:\n")
        try:  # strptime throws an exception if the input doesn't match the pattern
            start_time = datetime.strptime(start_time_input, "%Y-%m-%d-%H")
            end_time = datetime.strptime(end_time_input, "%Y-%m-%d-%H")
        except ValueError:
            print("The format doesn't match, try again!\n")
            continue
        if start_time >= end_time:
            print("Start time must be earlier than end time.\n")
            continue
        else:
            break
    return start_time, end_time
