import os


def gen_logfile(filename: str, location: str) -> str:
    """Converts the name of a file to a log file name, replacing the extension
    with '.log' and prepending the new path in a platform-independent way.

    :param filename: Original file name to be converted, usually a script name.
    :type filename: str

    :return: Log file path and name.
    :rtype: str
    """

    # Get the file name from the old path
    logfile = os.path.basename(filename)

    # Remove the old extension and add '.log' as the new one
    logfile = os.path.splitext(logfile)[0] + ".log"

    # Join the log file name with the new location
    logfile = os.path.join(location, logfile)

    return logfile
