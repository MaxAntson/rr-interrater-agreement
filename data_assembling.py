import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

warnings.filterwarnings("ignore")


def get_video_filenames(data_dir):
    for subdir, dirs, files in os.walk(data_dir):
        return np.sort(dirs)


def setup_dataframe(video_file_names, data_dir):
    rr_dataframe = pd.DataFrame(columns=(
        'file', 'considered_period', 'reviewer', 'distorted_period',
        'method_1', 'method_2', 'method_3', 'method_4'))
    for curr_video in video_file_names:
        for subdir, dirs, files in os.walk(os.path.join(data_dir, curr_video)):
            for file in files:
                # Get reviewer name
                reviewer_start = [i for i in range(len(file)) if file.startswith('_', i)][-1]
                reviewer_end = file.find('.csv')
                rr_dataframe = rr_dataframe.append(
                    {'file': curr_video, 'reviewer': file[reviewer_start + 1:reviewer_end]},
                    ignore_index=True)
    return rr_dataframe


def get_start_and_end_time_intersection_for_video(curr_video, data_dir):
    # Get intersection of start and end cut-offs for the video
    time_start, time_end = -np.inf, np.inf
    for subdir, dirs, files in os.walk(os.path.join(data_dir, curr_video)):
        for file in files:
            file_df = pd.read_csv(
                os.path.join(data_dir, curr_video, file), header=None, sep=';\t')

            # Deals with cases of annotations marked with "" at start and end
            if type(file_df[0][0]) == str and file_df[0][0][0] == '"':
                file_df[0] = [s[1:] for s in file_df[0]]

            # Change time cut-offs if needed
            if float(file_df[0][0]) > time_start:
                time_start = float(file_df[0][0])
            if float(file_df[0][len(file_df) - 1]) < time_end:
                time_end = float(file_df[0][len(file_df) - 1])
    return time_start, time_end


def method_1(file_df, time_start, time_end):
    time_stamps = np.array(file_df[0]).astype(float)
    readings = np.array(file_df[1])
    total_time = time_end - time_start

    # Get fractions at start and end
    fraction_before, fraction_after = 0, 0
    # Fraction at start
    if len(np.where(time_stamps < time_start)[0]) > 0:
        one_before = np.sort(np.where(time_stamps < time_start)[0])[
            len(np.where(time_stamps < time_start)[0]) - 1]
        # Locate first breath within time_start and time_end marked as 'N'
        i = one_before + 1
        while readings[i] != 'N' and i < len(readings):
            i += 1
        # Locate previous breath before time_start marked as 'N'
        while one_before >= 0 and readings[one_before] != 'N':
            one_before -= 1
        # Get fraction
        if one_before >= 0 and readings[one_before] == 'N':
            fraction_before = (time_stamps[i] - time_start) / (
                        time_stamps[i] - time_stamps[one_before])

    if len(np.where(time_stamps > time_end)[0]) > 0:
        one_after = np.sort(np.where(time_stamps > time_end)[0])[0]
        # Locate last breath within time_start and time_end marked as 'N'
        i = one_after - 1
        while readings[i] != 'N' and i >= 0:
            i -= 1
        # Locate later breath after time_end marked as 'N'
        while one_after < len(time_stamps) and readings[one_after] != 'N':
            one_after += 1
        # Get fraction
        if one_after < len(readings) and readings[one_after] == 'N':
            fraction_after = (time_end - time_stamps[i]) / (
                        time_stamps[one_after] - time_stamps[i])

    # Cut off readings and time stamps using time_start and time_end
    after = np.where(time_stamps >= time_start)[0]
    before = np.where(time_stamps <= time_end)
    all_stamps = np.sort(np.intersect1d(before, after))
    readings = readings[all_stamps]

    # Count total breaths (=number of 'N' - 1 + fractions)
    total_breaths = len(
        np.where(readings == 'N')[0]) + fraction_before + fraction_after - 1
    return (total_breaths / total_time) * 60


def method_2(file_df, time_start, time_end):
    time_stamps = np.array(file_df[0]).astype(float)
    readings = np.array(file_df[1])
    total_time = time_end - time_start

    # Get fractions at start and end
    fraction_before = 0
    fraction_after = 0
    if len(np.where(time_stamps < time_start)[0]) > 0:
        one_before = np.sort(np.where(time_stamps < time_start)[0])[
            len(np.where(time_stamps < time_start)[0]) - 1]
        # Locate first breath within time_start and time_end marked as 'N' or 'U'
        i = one_before + 1
        while readings[i] not in ['N', 'U'] and i < len(readings):
            i += 1
        # Locate previous breath before time_start marked as 'N' or 'U'
        while one_before >= 0 and readings[one_before] not in ['N', 'U']:
            one_before -= 1
        # Get fraction
        if one_before >= 0 and readings[one_before] in ['N', 'U']:
            fraction_before = (time_stamps[i] - time_start) / (
                        time_stamps[i] - time_stamps[one_before])

    if len(np.where(time_stamps > time_end)[0]) > 0:
        one_after = np.sort(np.where(time_stamps > time_end)[0])[0]
        # Locate last breath within time_start and time_end marked as 'N' or 'U'
        i = one_after - 1
        while readings[i] not in ['N', 'U'] and i >= 0:
            i -= 1
        # Locate later breath after time_end marked as 'N' or 'U'
        while one_after < len(time_stamps) and readings[one_after] not in ['N', 'U']:
            one_after += 1
        # Get fraction
        if one_after < len(readings) and readings[one_after] in ['N', 'U']:
            fraction_after = (time_end - time_stamps[i]) / (
                        time_stamps[one_after] - time_stamps[i])

    # Cut off readings and time stamps using time_start and time_end
    after = np.where(time_stamps >= time_start)[0]
    before = np.where(time_stamps <= time_end)
    all_stamps = np.sort(np.intersect1d(before, after))
    readings = readings[all_stamps]

    # Count total breaths (=number of 'N' and 'U' - 1 + fractions)
    total_breaths = len(
        np.where(readings == 'N')[0]) + fraction_before + fraction_after - 1
    total_breaths += len(np.where(readings == 'U')[0])
    return (total_breaths / total_time) * 60


def method_3(file_df, time_start, time_end):
    time_stamps = np.array(file_df[0]).astype(float)
    readings = np.array(file_df[1])

    # Find distorted and clean sections
    distorted = np.where(readings == 'x')[0]
    bad_parts = []
    for i in range(len(distorted)):
        bad_parts.append(distorted[i])
        bad = 1
        current_location = distorted[i] + 1
        while bad == 1 and current_location < len(readings):
            # Keep adding to 'bad' (i.e. distorted) indices until we reach an 'N'
            # followed by an 'N' or a 'U'
            if readings[current_location] == 'N' and current_location + 1 < len(
                    readings) and readings[current_location + 1] != 'x':
                bad = 0
            else:
                bad_parts.append(current_location)
                current_location += 1

        # Do the same as above but work backwards through the signal
        # (and don't worry about found N's being after an 'N' or a 'U', as per spec)
        bad = 1
        current_location = distorted[i] - 1
        while bad == 1 and current_location >= 0:
            if readings[current_location] == 'N':
                bad = 0
            else:
                bad_parts.append(current_location)
                current_location -= 1

    # Get good and bad sections
    bad_parts = np.unique(np.array(bad_parts))
    if len(bad_parts) > 0:
        good_parts = np.delete(np.arange(len(readings)), bad_parts)
    else:
        good_parts = np.arange(len(readings))

    # Group good sections together
    spl = [0] + [i for i in range(1, len(good_parts)) if
                 good_parts[i] - good_parts[i - 1] > 1] + [None]
    spl = spl[:len(spl)]
    split_up = [good_parts[b:e] for (b, e) in
                [(spl[i - 1], spl[i]) for i in range(1, len(spl))]]

    starts = np.array([split_up[i][0] for i in range(len(split_up))])
    ends = np.array([split_up[i][len(split_up[i]) - 1] for i in range(len(split_up))])

    # Get fractions
    fraction_before = 0
    fraction_after = 0
    if len(np.where(time_stamps < time_start)[0]) > 0:
        one_before = np.sort(np.where(time_stamps < time_start)[0])[
            len(np.where(time_stamps < time_start)[0]) - 1]

        # Go through signal and find the first 'N'. If there are any indices marked
        # as distorted in between, flag as rejected and don't calculate fraction
        i = one_before + 1
        reject = 0
        while readings[i] != 'N':
            i += 1
            if i not in good_parts:
                reject = 1

        # Work backwards through signal (before time_start) and find first 'N'.
        # If there are any indices marked as distorted in between, flag as rejected
        # and don't calculate fraction
        not_distorted = 1
        while one_before >= 0 and readings[one_before] != 'N':
            one_before -= 1
            if one_before >= 0 and one_before not in good_parts:
                not_distorted = 0
        # Get fraction if flags not changed
        if (one_before >= 0 and readings[one_before] == 'N' and
                not_distorted == 1 and reject == 0):
            fraction_before = (time_stamps[i] - time_start) / (
                        time_stamps[i] - time_stamps[one_before])

    if len(np.where(time_stamps > time_end)[0]) > 0:
        one_after = np.sort(np.where(time_stamps > time_end)[0])[0]

        # Go backwards signal and find the first 'N'. If there are any indices marked
        # as distorted in between, flag as rejected and don't calculate fraction
        i = one_after - 1
        reject = 0
        while readings[i] != 'N' and i >= 0:
            i -= 1
            if i not in good_parts:
                reject = 1

        # Work fowards through signal (after time_end) and find first 'N'. If there
        # are any indices marked as distorted in between, flag as rejected and don't
        # calculate fraction
        not_distorted = 1
        while one_after < len(readings) and readings[one_after] != 'N':
            one_after += 1
            if one_after < len(readings) and one_after not in good_parts:
                not_distorted = 0
        # Get fraction if flags not changed
        if (one_after < len(readings) and readings[one_after] == 'N'
                and not_distorted == 1 and reject == 0):
            fraction_after = (time_end - time_stamps[i]) / (
                        time_stamps[one_after] - time_stamps[i])

    # Cut off readings and time stamps using time_start and time_end
    after = np.where(time_stamps >= time_start)[0]
    before = np.where(time_stamps <= time_end)
    all_stamps = np.sort(np.intersect1d(before, after))

    # Shift 'starts' and 'ends' indices calculated at the start of the method to align
    # with new start after cut-offs applied
    difference_start = np.sort(after)[0]
    starts -= difference_start
    ends -= difference_start

    starts[starts < 0] = 0
    ends[ends < 0] = 0

    readings = readings[all_stamps]
    time_stamps = time_stamps[all_stamps]

    # Get numerator and denominator
    # Numerator calculated by looping through the 'good' groupings and summing number
    # of 'N's (and taking off 1, as before) Denominator calculated by looping through
    # 'good' groupings and finding total length of each grouping
    total_breaths = fraction_before + fraction_after
    total_time = 0
    for i in range(len(starts)):
        total_breaths += len(np.where(readings[starts[i]:ends[i]] == 'N')[0]) - 1
        total_time += time_stamps[np.max(
            np.array([0, np.min(np.array([ends[i], len(time_stamps) - 1]))]))] - \
            time_stamps[np.max(np.array(
                [0, np.min(np.array([starts[i], len(time_stamps) - 1]))]))]
    return (total_breaths / total_time) * 60
    
    
def method_4(file_df, time_start, time_end):
    time_stamps = np.array(file_df[0]).astype(float)
    readings = np.array(file_df[1])

    # Find distorted and clean sections
    distorted = np.where(readings == 'x')[0]
    bad_parts = []
    for i in range(len(distorted)):
        bad_parts.append(distorted[i])
        bad = 1
        current_location = distorted[i] + 1
        while bad == 1 and current_location < len(readings):
            # Keep adding to 'bad' (i.e. distorted) indices until we reach an 'N'
            # followed by an 'N' or a 'U'
            if readings[current_location] == 'N' and current_location + 1 < len(
                    readings) and readings[current_location + 1] != 'x':
                bad = 0
            else:
                bad_parts.append(current_location)
                current_location += 1
        # Do the same as above but work backwards through the signal (and don't worry
        # about found N's being after an 'N' or a 'U', as per spec)
        bad = 1
        current_location = distorted[i] - 1
        while bad == 1 and current_location >= 0:
            if readings[current_location] == 'N':
                bad = 0
            else:
                bad_parts.append(current_location)
                current_location -= 1

    # Get good and bad sections
    bad_parts = np.unique(np.array(bad_parts))
    if len(bad_parts) > 0:
        good_parts = np.delete(np.arange(len(readings)), bad_parts)
    else:
        good_parts = np.arange(len(readings))

    # Group good sections together
    spl = [0] + [i for i in range(1, len(good_parts)) if
                 good_parts[i] - good_parts[i - 1] > 1] + [None]
    spl = spl[:len(spl)]
    split_up = [good_parts[b:e] for (b, e) in
                [(spl[i - 1], spl[i]) for i in range(1, len(spl))]]

    starts = np.array([split_up[i][0] for i in range(len(split_up))])
    ends = np.array([split_up[i][len(split_up[i]) - 1] for i in range(len(split_up))])

    # Get fractions
    fraction_before = 0
    fraction_after = 0

    if len(np.where(time_stamps < time_start)[0]) > 0:
        one_before = np.sort(np.where(time_stamps < time_start)[0])[
            len(np.where(time_stamps < time_start)[0]) - 1]

        # If i not in good_parts, then i is in distorted and so this fraction should
        # not be considered
        i = one_before + 1
        reject = 0
        if i not in good_parts:
            reject = 1

        if one_before in good_parts and reject == 0:
            fraction_before = (time_stamps[i] - time_start) / (
                        time_stamps[i] - time_stamps[one_before])

    if len(np.where(time_stamps > time_end)[0]) > 0:
        one_after = np.sort(np.where(time_stamps > time_end)[0])[0]
        i = one_after - 1
        reject = 0
        if i not in good_parts:
            reject = 1

        if one_after in good_parts and reject == 0:
            fraction_after = (time_end - time_stamps[i]) / (
                        time_stamps[one_after] - time_stamps[i])

    after = np.where(time_stamps >= time_start)[0]
    before = np.where(time_stamps <= time_end)
    all_stamps = np.sort(np.intersect1d(before, after))

    difference_start = np.sort(after)[0]
    starts -= difference_start
    ends -= difference_start
    starts[starts < 0] = 0
    ends[ends < 0] = 0

    readings = readings[all_stamps]
    time_stamps = time_stamps[all_stamps]
    
    total_breaths = fraction_before + fraction_after
    total_time = 0
    for i in range(len(starts)):
        total_breaths += len(readings[starts[i]:ends[i]]) - 1
        total_time += time_stamps[np.max(
            np.array([0, np.min(np.array([ends[i], len(time_stamps) - 1]))]))] - \
            time_stamps[np.max(np.array(
                [0, np.min(np.array([starts[i], len(time_stamps) - 1]))]))]
    return (total_breaths / total_time) * 60, total_time
                
    
def run_and_output_analysis(data_dir, output_path="mc_rr_outputs.csv"):
    video_file_names = get_video_filenames(data_dir)
    rr_dataframe = setup_dataframe(video_file_names, data_dir)
    for curr_video in tqdm(video_file_names):
        time_start, time_end = get_start_and_end_time_intersection_for_video(curr_video, data_dir)

        # Loop through reviewers for current video 
        for subdir, dirs, files in os.walk(os.path.join(data_dir, curr_video)):
            for file in files:
                file_df = pd.read_csv(
                    os.path.join(data_dir, curr_video, file), header=None, sep=';\t')

                reviewer_start = [i for i in range(len(file)) if file.startswith('_', i)][-1]
                reviewer_end = file.find('.csv')

                # Deals with cases of annotations marked with "" at start and end
                if type(file_df[0][0]) == str and file_df[0][0][0] == '"':
                    file_df[0] = [i[1:] for i in file_df[0]]
                    file_df[1] = [i[:-1] for i in file_df[1]]
            
                index = np.intersect1d(
                    np.where(rr_dataframe['file'] == curr_video)[0],
                    np.where(rr_dataframe['reviewer'] == file[reviewer_start + 1:reviewer_end])[0])
                
                rr_dataframe['method_1'][index] = method_1(file_df, time_start, time_end)
                rr_dataframe['method_2'][index] = method_2(file_df, time_start, time_end)
                rr_dataframe['method_3'][index] = method_3(file_df, time_start, time_end)
                rr_dataframe['method_4'][index], total_time = method_4(
                    file_df, time_start, time_end)

                rr_dataframe['distorted_period'][index] = (time_end - time_start) - total_time
                rr_dataframe['considered_period'][index] = time_end - time_start
    rr_dataframe.to_csv(output_path, index=False)
