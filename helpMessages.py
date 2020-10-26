
get_started = """
\nWelcome to SAFT - Semi-Automatic Fluorescence Trace analysis
\n----------------------------
\nHere is some advice for novice users
\nLoad your data (traces in columns) from Excel. The first row should name the regions of interest (ROI), and each condition should be a separate sheet: 0.5 mM, 2 mM, 4 mM, etc. The ROIs do not have to be the same for each condition.
\nAdjust the baseline and find peaks automatically.
\nIn the 'Peak editing' window, turn on "Edit peaks with mouse". You can manually add peaks by left-clicking (and left-click on existing peaks to remove). Histogram should update as you go. Your clicks are locked to the data. You can do this manually for every trace if you like.
\nBetter: the "Extract peaks for all ROIs" button will open a dialog that uses the positions of the peaks from the 4 mM "mean" trace to get all the peaks from every other ROI. You can optionally blacklist ROIs from analysis that have a bad SNR. You can also select a region around each peak for the search.
\nSave the peaks and also the automatically baselined traces from the File menu or buttons. Peaks are sorted in SNR order.
\nHistograms (summed for each ROI over conditions or separated) can be saved as well.
\nIf traces consist of grouped responses to repeated stimuli, these can be batch analysed with "Extract group stats".
\nThe statistics of peaks can be analysed in the separate module histogramDialog.py
"""
