import datetime

def split_time_range_into_intervals(
    from_datetime, to_datetime, n):

    total_duration = to_datetime - from_datetime
    interval_length = total_duration / n

    # Generate the interval.
    intervals = []
    for i in range(n):
        interval_start = from_datetime + (i * interval_length)
        interval_end = from_datetime + ((i + 1) * interval_length)
        if i + 1 != n:
            # Subtract 1 microsecond from the end of each interval to avoid overlapping.
            interval_end = interval_end - datetime.timedelta(minutes=1)

        intervals.append((interval_start, interval_end))

    return intervals