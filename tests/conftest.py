'''Shared pytest fixtures/config.

Forces the non-interactive 'Agg' matplotlib backend for the whole test
session. Without this, matplotlib picks up whatever interactive backend
(e.g. TkAgg) happens to be available in the environment, and plotting code
paths that call plt.show() intermittently raise TclError when many figures
are created across a test run. Tests only need plots to render to a
savefig() buffer/file, never to an actual screen, so Agg is correct here
regardless of what a real user's environment resolves to.
'''
import matplotlib
matplotlib.use('Agg')
