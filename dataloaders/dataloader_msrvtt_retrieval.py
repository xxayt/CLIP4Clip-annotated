from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from dataloaders.rawvideo_util import RawVideoExtractor

class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask



class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,  # 包含训练数据的videoID和其他元数据的CSV文件路径
            json_path,  # 包含video和caption配对的JSON文件路径
            features_path,  # video特征数据的路径，通常是已提取的特征或原始video路径
            tokenizer,  # 用于处理caption文本的分词器
            max_words=30,  # 每个caption允许的最大词数
            feature_framerate=1.0,  # 每秒提取的特征帧率
            max_frames=100,  # 每个video允许的最大帧数
            unfold_sentences=False,  # 是否展开所有videocaption对，False表示一个video配对多个caption
            image_resolution=224,  # 图像帧的分辨率
            frame_order=0,  # 帧的顺序，0: 正常顺序；1: 反向顺序；2: 随机顺序
            slice_framepos=0,  # 截取帧的方式，0: 从开头截取；1: 从尾部截取；2: 均匀截取
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0  # 初始化样本数量
        if self.unfold_sentences:  # 一个caption对应一个sample
            train_video_ids = list(self.csv['video_id'].values)  # [video1, video2, video3,...]
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                '''
                itm = {
                    "caption": "a cartoon animals runs through an ice cave in a video game",
                    "video_id": "video2960",
                    "sen_id": 0
                }
                '''
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)  # sample number = caption number，总计20w条sen_id
        else:  # 一个video对应一个sample
            num_sentences = 0
            self.sentences = defaultdict(list)  # 新建一个value为list的dict，用于存储每个video的caption列表
            s_video_id_set = set()  # set for storing all video_ids
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])  # {video1: [caption1, caption2,...], video2: [caption1, caption2,...],...}
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)  # 新建一个value为list的dict，用于存储每个url_posfix的video列表
            for itm in self.data['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)  # sample number = video number，总计9k/7k条video_id

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    # 给定video_id和caption，返回对应的caption文本的token id、mask、segment id、video_id
    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:  # unfold_sentences (每个caption都对应一个sample)
                words = self.tokenizer.tokenize(caption)
            else:  # fold_sentences (每个video才对应一个sample)
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1  # 12-1=11
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)  # token id list
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words  # eg: [4233, 40114, 33139, 779, 320, 886, 6332, 633, 518, 0, 0, 0]
            assert len(input_mask) == self.max_words  # eg: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
            assert len(segment_ids) == self.max_words  # 相当于 [0] * max_words
            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]  # 在video_id对应的caption列表中随机选取一个caption
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)  # 记录有效帧长度 eg: [1, 12]
        max_video_length = [0] * len(choice_video_ids)  # 记录实际帧长度
        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)  # 记录视频特征

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W （L=12, T=1）
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:  # uniformly extract frames
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice
                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)
                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length
        return video, video_mask  # [1, 12, 1, 3, 224, 224], [1, 12]

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask
