import os
import cv2
import h5py
import torch
import numpy as np
from script import utils
from torch.utils.data import Dataset


class test_SEID_sevfi(Dataset):
    def __init__(self, data_path, num_bins, num_skip, num_insert, delta_ts=0.0083):
        '''
        Parameters
        ----------
        data_path: str
            path of target data.
        num_skip: int
            the number of skip frames.
        num_insert: int
            the number of insert frames.
        '''
        self.data_path = data_path
        self.events_path = self.data_path + '/events/'
        self.images_path = self.data_path + '/images/'
        self.events_list = utils.get_filename(self.events_path, '.h5')
        self.images_list = utils.get_filename(self.images_path, '.png')
        self.image_timestamps = np.loadtxt(os.path.join(self.images_path, 'timestamp.txt'))
        self.num_bins = num_bins
        self.num_skip = num_skip
        self.num_insert = num_insert
        image = cv2.imread(os.path.join(self.images_path, self.images_list[0]), cv2.IMREAD_COLOR)
        self.img_size = image.shape
        self.voxel_grid = utils.VoxelGrid(self.num_bins, self.img_size[0], self.img_size[1], normalize=True)
        self.delta_ts = delta_ts  # for Image of Warped Events

    def events_to_voxel_grid(self, x, y, p, t, device: str = 'cpu'):
        if len(t) <= 10:
            return self.voxel_grid.voxel_grid
        elif len(t) > 10:
            t = (t - t[0]).astype('float32')
            t = (t / t[-1])
            x = x.astype('float32')
            y = y.astype('float32')
            pol = p.astype('float32')
            return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def __len__(self):
        self.len_all_data = len(self.events_list)
        self.len = (self.len_all_data - 1) // (self.num_skip + 1)
        return self.len

    def __getitem__(self, index):
        index_0 = index * (self.num_skip + 1)
        index_1 = index_0 + (self.num_skip + 1)

        ##### load images #####
        image_0 = cv2.imread(os.path.join(self.images_path, self.images_list[index_0]), cv2.IMREAD_COLOR)
        image_1 = cv2.imread(os.path.join(self.images_path, self.images_list[index_1]), cv2.IMREAD_COLOR)
        TS = []
        TS.append(self.image_timestamps[index_0])
        time_span = self.image_timestamps[index_1] - self.image_timestamps[index_0]
        interval = time_span / (self.num_insert + 1)
        for i in range(self.num_insert):
            TS.append(self.image_timestamps[index_0] + interval * (i + 1))
        TS.append(self.image_timestamps[index_1])

        ##### load events for interoplation #####
        x, y, p, t = np.empty(shape=0), np.empty(shape=0), np.empty(shape=0), np.empty(shape=0)
        event_index_0 = index_0 - 1
        event_index_1 = index_1 + 1
        for e in range(event_index_0, event_index_1, 1):
            e_start = int(self.events_list[0][:-3])
            if e < e_start or e >= len(self.events_list):
                pass
            else:
                tmp_h5_path = self.events_path + '0' + self.images_list[e][:-4] + '.h5'
                h5f = h5py.File(tmp_h5_path, "r")
                events = dict()
                for dset_str in ['p', 'x', 'y', 't']:
                    events[dset_str] = h5f['{}'.format(dset_str)]
                    events[dset_str] = np.asarray(events[dset_str])
                p = np.hstack((p, events['p']))
                t = np.hstack((t, events['t']))
                x = np.hstack((x, events['x']))
                y = np.hstack((y, events['y']))

        ##### crop events #####
        # Cropping (+- 2 for safety reasons)
        x_mask = (x >= 0) & (x < self.img_size[1])
        y_mask = (y >= 0) & (y < self.img_size[0])
        mask_combined = x_mask & y_mask
        p = p[mask_combined]
        t = t[mask_combined]
        x = x[mask_combined]
        y = y[mask_combined]
        ##### IWE generate #####
        iwe = []
        for ts in TS:
            t_start = ts - self.delta_ts
            t_end = ts + self.delta_ts
            # np.add.at for single channel event frames
            event_iwe = utils.filter_events_by_time(x, y, p, t, t_start, t_end)
            x_img = event_iwe[0].astype(np.int32)
            y_img = event_iwe[1].astype(np.int32)
            p_img = event_iwe[2].astype(np.int8)
            p_img[p_img == 0] = -1
            event_img = np.zeros(self.img_size[0] * self.img_size[1], dtype=np.float32)
            np.add.at(event_img, y_img * self.img_size[1] + x_img, p_img)
            event_img = event_img.reshape([self.img_size[0], self.img_size[1]])
            iwe.append(event_img)
        iwe = iwe[1:-1]
        ##### events to voxel grid #####
        # initial list
        eframes_t0 = []
        eframes_t1 = []
        ts_0 = TS[0]
        ts_1 = TS[-1]

        for i in range(self.num_insert):
            ts_t = TS[i + 1]
            event_0t = utils.filter_events_by_time(x, y, p, t, ts_0, ts_t)
            event_t1 = utils.filter_events_by_time(x, y, p, t, ts_t, ts_1)
            voxel_t1 = self.events_to_voxel_grid(event_t1[0], event_t1[1], event_t1[2], event_t1[3])
            x_t0, y_t0, p_t0, t_t0 = utils.reverse_events(event_0t[0], event_0t[1], event_0t[2], event_0t[3])
            voxel_t0 = self.events_to_voxel_grid(x_t0, y_t0, p_t0, t_t0)
            eframes_t1.append(voxel_t1.detach().numpy())
            eframes_t0.append(voxel_t0.detach().numpy())

        ##### calculate weight #####
        weight = []
        ts_0 = TS[0]
        ts_1 = TS[-1]
        dt_01 = ts_1 - ts_0
        for i in range(self.num_insert):
            ts_t = TS[i+1]
            dt_t0 = ts_t - ts_0
            weight.append(dt_t0 / dt_01)

        ##### list to array #####
        TS = np.array(TS)
        eframes_t1 = np.array(eframes_t1)
        eframes_t0 = np.array(eframes_t0)
        weight = np.array(weight)
        iwe = np.array(iwe)

        ##### get data dict #####
        sample = {}
        sample['timestamps'] = TS
        sample['image_0'] = image_0
        sample['image_1'] = image_1
        sample['eframes_t1'] = eframes_t1
        sample['eframes_t0'] = eframes_t0
        sample['weight'] = weight
        sample['iwe'] = iwe

        return sample

class test_DSEC_sevfi(Dataset):
    def __init__(self, data_path, num_bins, num_skip, num_insert, delta_ts=25000):
        '''
        Parameters
        ----------
        data_path: str
            path of target data.
        num_skip: int
            the number of skip frames.
        num_insert: int
            the number of insert frames.
        '''
        self.data_path = data_path
        self.events_path = self.data_path + '/events/'
        self.images_path = self.data_path + '/images/'
        self.events_list = utils.get_filename(self.events_path, '.h5')
        self.images_list = utils.get_filename(self.images_path, '.png')
        self.image_timestamps = np.loadtxt(os.path.join(self.images_path, 'timestamp.txt'))
        self.num_bins = num_bins
        self.num_skip = num_skip
        self.num_insert = num_insert
        image = cv2.imread(os.path.join(self.images_path, self.images_list[0]), cv2.IMREAD_COLOR)
        self.img_size = image.shape
        self.voxel_grid = utils.VoxelGrid(self.num_bins, self.img_size[0], self.img_size[1], normalize=True)
        self.delta_ts = delta_ts  # for Image of Warped Events

    def events_to_voxel_grid(self, x, y, p, t, device: str = 'cpu'):
        if len(t) <= 10:
            return self.voxel_grid.voxel_grid
        elif len(t) > 10:
            t = (t - t[0]).astype('float32')
            t = (t / t[-1])
            x = x.astype('float32')
            y = y.astype('float32')
            pol = p.astype('float32')
            return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def __len__(self):
        self.len_all_data = len(self.image_timestamps)
        self.len = (self.len_all_data - 1) // (self.num_skip + 1)
        return self.len

    def __getitem__(self, index):
        # index += 1  # remove the first image
        index_0 = index * (self.num_skip + 1)
        index_1 = index_0 + (self.num_skip + 1)

        ##### load images #####
        image_0 = cv2.imread(os.path.join(self.images_path, self.images_list[index_0]), cv2.IMREAD_COLOR)
        image_1 = cv2.imread(os.path.join(self.images_path, self.images_list[index_1]), cv2.IMREAD_COLOR)
        TS = []
        TS.append(self.image_timestamps[index_0])
        time_span = self.image_timestamps[index_1] - self.image_timestamps[index_0]
        interval = time_span / (self.num_insert + 1)
        for i in range(self.num_insert):
            TS.append(self.image_timestamps[index_0] + interval * (i + 1))
        TS.append(self.image_timestamps[index_1])

        ##### load events for interoplation #####
        x, y, p, t = np.empty(shape=0), np.empty(shape=0), np.empty(shape=0), np.empty(shape=0)
        event_index_0 = index_0 - 1
        event_index_1 = index_1 + 1
        for e in range(event_index_0, event_index_1, 1):
            events = dict()
            if e < 0 or e >= len(self.events_list):
                pass
            else:
                h5f = h5py.File(os.path.join(self.events_path, self.events_list[e]), "r")
                for dset_str in ['p', 'x', 'y', 't']:
                    events[dset_str] = h5f['{}'.format(dset_str)]
                    events[dset_str] = np.asarray(events[dset_str])
                p = np.hstack((p, events['p']))
                t = np.hstack((t, events['t']))
                x = np.hstack((x, events['x']))
                y = np.hstack((y, events['y']))

        ##### crop events #####
        # Cropping (+- 2 for safety reasons)
        x_mask = (x >= 0) & (x < self.img_size[1])
        y_mask = (y >= 0) & (y < self.img_size[0])
        mask_combined = x_mask & y_mask
        p = p[mask_combined]
        t = t[mask_combined]
        x = x[mask_combined]
        y = y[mask_combined]
        ##### IWE generate #####
        iwe = []
        for ts in TS:
            t_start = ts - self.delta_ts
            t_end = ts + self.delta_ts
            # np.add.at for single channel event frames
            event_iwe = utils.filter_events_by_time(x, y, p, t, t_start, t_end)
            x_img = event_iwe[0].astype(np.int32)
            y_img = event_iwe[1].astype(np.int32)
            p_img = event_iwe[2].astype(np.int8)
            p_img[p_img == 0] = -1
            event_img = np.zeros(self.img_size[0] * self.img_size[1], dtype=np.float32)
            np.add.at(event_img, y_img * self.img_size[1] + x_img, p_img)
            event_img = event_img.reshape([self.img_size[0], self.img_size[1]])
            iwe.append(event_img)
        iwe = iwe[1:-1]
        ##### events to voxel grid #####
        # initial list
        eframes_t0 = []
        eframes_t1 = []
        ts_0 = TS[0]
        ts_1 = TS[-1]

        ##### voxel01\10 #####
        event_01 = utils.filter_events_by_time(x, y, p, t, ts_0, ts_1)
        x_10, y_10, p_10, t_10 = utils.reverse_events(event_01[0], event_01[1], event_01[2], event_01[3])
        voxel_01 = self.events_to_voxel_grid(event_01[0], event_01[1], event_01[2], event_01[3])

        for i in range(self.num_insert):
            ts_t = TS[i + 1]
            event_0t = utils.filter_events_by_time(x, y, p, t, ts_0, ts_t)
            event_t1 = utils.filter_events_by_time(x, y, p, t, ts_t, ts_1)
            voxel_t1 = self.events_to_voxel_grid(event_t1[0], event_t1[1], event_t1[2], event_t1[3])
            x_t0, y_t0, p_t0, t_t0 = utils.reverse_events(event_0t[0], event_0t[1], event_0t[2], event_0t[3])
            voxel_t0 = self.events_to_voxel_grid(x_t0, y_t0, p_t0, t_t0)
            eframes_t1.append(voxel_t1.detach().numpy())
            eframes_t0.append(voxel_t0.detach().numpy())

        ##### calculate weight #####
        weight = []
        ts_0 = TS[0]
        ts_1 = TS[-1]
        dt_01 = ts_1 - ts_0
        for i in range(self.num_insert):
            ts_t = TS[i+1]
            dt_t0 = ts_t - ts_0
            weight.append(dt_t0 / dt_01)

        ##### list to array #####
        TS = np.array(TS)
        eframes_t1 = np.array(eframes_t1)
        eframes_t0 = np.array(eframes_t0)
        weight = np.array(weight)
        iwe = np.array(iwe)

        ##### get data dict #####
        sample = {}
        sample['timestamps'] = TS
        sample['image_0'] = image_0
        sample['image_1'] = image_1
        sample['eframes_t1'] = eframes_t1
        sample['eframes_t0'] = eframes_t0
        sample['weight'] = weight
        sample['iwe'] = iwe
        return sample

class test_MVSEC_sevfi(Dataset):
    def __init__(self, data_path, num_bins, num_skip, num_insert, delta_ts=0.016):
        '''
        Parameters
        ----------
        data_path: str
            path of target data.
        num_skip: int
            the number of skip frames.
        num_insert: int
            the number of insert frames.
        '''
        self.data_path = data_path
        self.events_path = self.data_path + '/events/'
        self.images_path = self.data_path + '/images/'
        self.events_list = utils.get_filename(self.events_path, '.h5')
        self.images_list = utils.get_filename(self.images_path, '.png')
        self.image_timestamps = np.loadtxt(os.path.join(self.images_path, 'timestamp.txt'))
        self.num_bins = num_bins
        self.num_skip = num_skip
        self.num_insert = num_insert
        image = cv2.imread(os.path.join(self.images_path, self.images_list[0]), cv2.IMREAD_COLOR)
        self.img_size = image.shape
        self.voxel_grid = utils.VoxelGrid(self.num_bins, self.img_size[0], self.img_size[1], normalize=True)
        self.delta_ts = delta_ts  # for Image of Warped Events

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        if len(t) <= 10:
            return self.voxel_grid.voxel_grid
        elif len(t) > 10:
            t = (t - t[0]).astype('float32')
            t = (t/t[-1])
            x = x.astype('float32')
            y = y.astype('float32')
            pol = p.astype('float32')
            return self.voxel_grid.convert(
                    torch.from_numpy(x),
                    torch.from_numpy(y),
                    torch.from_numpy(pol),
                    torch.from_numpy(t))

    def __len__(self):
        self.len_all_data = len(self.image_timestamps)
        self.len = (self.len_all_data - 1)//(self.num_skip + 1)
        return self.len

    def __getitem__(self, index):
        # index += 1  # remove the first image
        index_0 = index * (self.num_skip + 1)
        index_1 = index_0 + (self.num_skip + 1)

        ##### load images #####
        image_0 = cv2.imread(os.path.join(self.images_path, self.images_list[index_0]), cv2.IMREAD_COLOR)
        image_1 = cv2.imread(os.path.join(self.images_path, self.images_list[index_1]), cv2.IMREAD_COLOR)
        TS = []
        TS.append(self.image_timestamps[index_0])
        time_span = self.image_timestamps[index_1] - self.image_timestamps[index_0]
        interval = time_span / (self.num_insert + 1)
        for i in range(self.num_insert):
            TS.append(self.image_timestamps[index_0] + interval * (i + 1))
        TS.append(self.image_timestamps[index_1])

        ##### load events for interoplation #####
        x, y, p, t = np.empty(shape=0), np.empty(shape=0), np.empty(shape=0), np.empty(shape=0)
        event_index_0 = index_0 - 1
        event_index_1 = index_1 + 1
        for e in range(event_index_0, event_index_1, 1):
            events = dict()
            if e < 0 or e >= len(self.events_list):
                pass
            else:
                h5f = h5py.File(os.path.join(self.events_path, self.events_list[e]), "r")
                for dset_str in ['p', 'x', 'y', 't']:
                    events[dset_str] = h5f['{}'.format(dset_str)]
                    events[dset_str] = np.asarray(events[dset_str])
                p = np.hstack((p, events['p']))
                t = np.hstack((t, events['t']))
                x = np.hstack((x, events['x']))
                y = np.hstack((y, events['y']))

        ##### IWE generate #####
        iwe = []
        for ts in TS:
            t_start = ts-self.delta_ts
            t_end = ts+self.delta_ts
            # np.add.at for single channel event frames
            event_iwe = utils.filter_events_by_time(x, y, p, t, t_start, t_end)
            x_img = event_iwe[0].astype(np.int32)
            y_img = event_iwe[1].astype(np.int32)
            p_img = event_iwe[2].astype(np.int8)
            p_img[p_img == 0] = -1
            event_img = np.zeros(self.img_size[0] * self.img_size[1], dtype=np.float32)
            np.add.at(event_img, y_img * self.img_size[1] + x_img, p_img)
            event_img = event_img.reshape([self.img_size[0], self.img_size[1]])
            iwe.append(event_img)
        iwe = iwe[1:-1]
        ##### events to voxel grid #####
        # initial list
        eframes_t0 = []
        eframes_t1 = []
        ts_0 = TS[0]
        ts_1 = TS[-1]

        for i in range(self.num_insert):
            ts_t = TS[i+1]
            event_0t = utils.filter_events_by_time(x, y, p, t, ts_0, ts_t)
            event_t1 = utils.filter_events_by_time(x, y, p, t, ts_t, ts_1)
            voxel_t1 = self.events_to_voxel_grid(event_t1[0], event_t1[1], event_t1[2], event_t1[3])
            x_t0, y_t0, p_t0, t_t0 = utils.reverse_events(event_0t[0], event_0t[1], event_0t[2], event_0t[3])
            voxel_t0 = self.events_to_voxel_grid(x_t0, y_t0, p_t0, t_t0)
            eframes_t1.append(voxel_t1.detach().numpy())
            eframes_t0.append(voxel_t0.detach().numpy())

        ##### calculate weight #####
        weight = []
        ts_0 = TS[0]
        ts_1 = TS[-1]
        dt_01 = ts_1 - ts_0
        for i in range(self.num_insert):
            ts_t = TS[i+1]
            dt_t0 = ts_t - ts_0
            weight.append(dt_t0 / dt_01)

        ##### list to array #####
        TS = np.array(TS)
        eframes_t1 = np.array(eframes_t1)
        eframes_t0 = np.array(eframes_t0)
        weight = np.array(weight)
        iwe = np.array(iwe)

        ##### get data dict #####
        sample = {}
        sample['timestamps'] = TS
        sample['image_0'] = image_0
        sample['image_1'] = image_1
        sample['eframes_t1'] = eframes_t1
        sample['eframes_t0'] = eframes_t0
        sample['weight'] = weight
        sample['iwe'] = iwe

        return sample

