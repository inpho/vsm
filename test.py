import pickle
print "importing data"
with open('/tmp/update_args_20.pyo', 'rb') as dumpfile:
    data = pickle.load(dumpfile)


print "testing function"
from _cgs_update import cgs_update_short_char as update
import line_profiler
profile = line_profiler.LineProfiler(update)
profile.runcall(update, *data)
print "printing stats"
profile.print_stats()

