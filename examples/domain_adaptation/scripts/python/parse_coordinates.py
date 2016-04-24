import os_helper as osh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_sorted_arg(row, sim):
    if sim:
        return np.argsort(row)[::-1]
    else:
        return np.argsort(row)


def create_score_mat(qu, db):
    score_mat = np.ones((qu.shape[0], qu.shape[0]))
    for i, qu_point in enumerate(qu):
        for j, db_point in enumerate(db):
            score_mat[i, j] = np.linalg.norm(qu_point - db_point)
        if i % 50 == 0:
            print i
    return np.apply_along_axis(lambda row: row / np.linalg.norm(row), 1, score_mat)


def pr_recall(score_mat, sim=True, im_range=3, threshold=.045):
    true_pos, false_neg, false_pos = 0, 0, 0
    for i, row in enumerate(score_mat):
        closest_arg = np.argmax(row) if sim else np.argmin(row) #get_sorted_arg(row, sim)
        if (sim and row[closest_arg] > threshold) or (not sim and row[closest_arg] < threshold):
            range_arr = range(i - im_range, i + im_range + 1)
            if closest_arg in range_arr:
                true_pos += 1
            else:
                false_pos += 1
        else:
            false_neg += 1

    pr_denom = float(true_pos + false_pos)
    recall_denom = float(true_pos + false_neg)
    pr = (true_pos / pr_denom) if pr_denom > 0 else 0
    recall = (true_pos / recall_denom) if recall_denom > 0 else 0
    return pr, recall


def vals_around_diag(score_mat, sim=True, k=3, diag=3):
    values_inside, values_outside = 0, 0
    for i, row in enumerate(score_mat):
        sorted_args = get_sorted_arg(row,sim)[:k]
        range_arr = range(i - diag, i + diag +1)
        for min_arg in sorted_args:
            if min_arg in range_arr:
                values_inside += 1
            else:
                values_outside += 1
    total_pts = values_inside + values_outside
    return values_inside / float(total_pts)


def main():
    score_data = [
        #['untrained_places205CNN_iter_300000_upgraded.caffemodel_nordland_fc7fc7_p_cos_sim.npy', #0.03, 0.011, True]]
        #['untrained_places205CNN_iter_300000_upgraded.caffemodel_nordland_cos_summer_conv3_norm_winter_conv3_norm.npy',
        #0.0225, 0.0105, True]]
        #['untrained_places205CNN_iter_300000_upgraded.caffemodel_nordland_cos_summer_conv3_norm_spring_conv3_norm.npy',
        #0.0233060606061, 0.0096, True]]
        #['untrained_places205CNN_iter_300000_upgraded.caffemodel_nordland_cos_summer_conv3_norm_fall_conv3_norm.npy',
        #0.0314, 0.012, True]]
        #['untrained_places205CNN_iter_300000_upgraded.caffemodel_nordland_euc_conv3_conv3_p.npy',
        #30.006, 0.00966666666667, False]]
        #['nordland_only_snapshots_iter_140000.caffemodel_nordland_cos_sim.npy',
        #0.0159909090909, 0.0115, True]]
        #['nordland_only_snapshots_iter_140000.caffemodel_nordland_euc_conv3_conv3_p.npy',
        #0.006, 0.0095, False]]
        #['nordland_only_snapshots_iter_60000.caffemodel_nordland_euc_summer_conv3_winter_conv3.npy',
        #0.0063, .01046, False]]
        #['nordland_only_snapshots_iter_60000.caffemodel_nordland_euc_summer_conv3_fall_conv3.npy',
        #0.00375100671141, .01, False]]
        #['nordland_only_snapshots_iter_120000_10_margin.caffemodel_nordland_euc_fc8_n_fc8_n_p.npy',
        #.06, 1.5, False]]
        #['nordland_only_snapshots_iter_120000_10_margin.caffemodel_nordland_euc_fc7_fc7_p.npy',
        #.00075, .0057, False]]
        #['nordland_only_snapshots_iter_120000.caffemodel_nordland_euc_summer_conv3_winter_conv3.npy',
        #0.00638832214765, .01039, False]]
        #['nordland_only_snapshots_iter_120000.caffemodel_nordland_euc_summer_conv3_spring_conv3.npy',
        #.00452, .0108, False]]
        #['nordland_only_snapshots_iter_120000.caffemodel_nordland_euc_summer_conv3_fall_conv3.npy',
        #.004, .0087, False]]
        #['untrained_places205CNN_iter_300000_upgraded.caffemodel_freiburg_cos_summer_conv3_norm_winter_conv3_norm.npy',
        #0.0485503355705, 0.0248862663844, True]]
        #['freiburg_only_snapshots_iter_5000.caffemodel_freiburg_euc_summer_conv3_winter_conv3.npy',
        #.0202, .0275, False]]
        ['freiburg_only_snapshots_iter_120000.caffemodel_freiburg_euc_summer_conv3_winter_conv3.npy', .0202, .0275, False]]
        #['final.npy', #.005, .0108, False]]

    for data in score_data:
        print 'processing: {0}'.format(data[0])
        score_mat = np.load(results_folder + data[0])
        pr_recal_list = []
        score_mat = np.apply_along_axis(lambda row: row / np.linalg.norm(row), 1, score_mat)
        #score_mat = np.apply_along_axis(lambda row: row / np.max(row), 1, score_mat)
        #score_mat = np.divide(np.subtract(score_mat, np.min(score_mat)), np.max(score_mat) - np.min(score_mat))
        print score_mat
        values = np.linspace(data[1], data[2], 150)
        for i, value in enumerate(values):
            pr_recal_result = pr_recall(score_mat, threshold=value, sim=data[-1])
            pr_recal_list.append(pr_recal_result)
            print i, value, pr_recal_result
        np.save(results_folder + 'pr_recall_' + data[0].replace('.npy', ''), np.array(pr_recal_list))
    return


if __name__ == '__main__':
    caffe_root = osh.get_env_var('CAFFE_ROOT')
    coord_file_path = caffe_root + '/data/domain_adaptation_data/images/coordinates'
    results_folder = caffe_root + '/data/domain_adaptation_data/results/freiburg/'
    main()
