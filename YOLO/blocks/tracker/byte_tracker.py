import numpy as np

from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    """
    BYTETracker - 基于字节跟踪算法的多目标跟踪器
    用于在视频序列中对检测到的目标进行跟踪和身份管理
    """
    
    def __init__(self, track_thresh, track_buffer, match_thresh, mot20, frame_rate=30):
        """
        初始化BYTETracker
        
        参数:
            track_thresh: 高分检测的置信度阈值，用于第一阶段关联
            track_buffer: 轨迹缓冲区大小（帧数），决定丢失轨迹保留多久
            match_thresh: 匹配距离阈值，用于确定是否关联检测框和轨迹
            mot20: 布尔值，是否使用MOT20数据集的设置
            frame_rate: 视频帧率，默认30fps
        """
        self.tracked_stracks = []  # type: list[STrack]  # 当前活跃的轨迹列表
        self.lost_stracks = []  # type: list[STrack]      # 丢失但未删除的轨迹列表
        self.removed_stracks = []  # type: list[STrack]   # 已删除的轨迹列表

        self.frame_id = 0                                   # 当前处理的帧序号
        self.det_thresh = track_thresh                      # 检测置信度阈值
        self.track_thresh = track_thresh                    # 跟踪置信度阈值
        # 根据帧率计算缓冲区大小，保证在不同帧率下的一致性
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size               # 轨迹最长丢失时间
        self.kalman_filter = KalmanFilter()                 # 初始化卡尔曼滤波器用于轨迹预测
        self.match_thresh = match_thresh                    # 匹配阈值
        self.mot20 = mot20                                  # MOT20数据集标志

    def update(self, output_results):
        """
        根据检测结果更新轨迹状态
        
        参数:
            output_results: 检测结果，形状为 (N, 4+) 或 (N, 6)
                           [x1, y1, x2, y2, score] 或 [x1, y1, x2, y2, score1, score2]
        
        返回:
            output_stracks: 当前帧的活跃轨迹列表
        """
        self.frame_id += 1
        activated_starcks = []    # 当前帧新激活的轨迹
        refind_stracks = []       # 当前帧重新找到的轨迹（从丢失列表恢复）
        lost_stracks = []         # 当前帧新丢失的轨迹
        removed_stracks = []      # 当前帧需要删除的轨迹

        # 步骤 0: 提取检测框和置信度分数
        if output_results.shape[1] == 5:
            # 单置信度格式: [x1, y1, x2, y2, score]
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            # 双置信度格式: [x1, y1, x2, y2, score1, score2]
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]  # 置信度相乘
            bboxes = output_results[:, :4]  # x1y1x2y2格式

        # 根据置信度分级检测框
        remain_inds = scores > self.track_thresh             # 高置信度检测框索引
        inds_low = scores > 0.1                             # 低置信度检测框索引（下限）
        inds_high = scores < self.track_thresh              # 低置信度检测框索引（上限）
        inds_second = np.logical_and(inds_low, inds_high)   # 第二阶段用的检测框索引

        # 分离高置信度和低置信度检测框
        dets_second = bboxes[inds_second]     # 低置信度检测框
        dets = bboxes[remain_inds]            # 高置信度检测框
        scores_keep = scores[remain_inds]     # 高置信度分数
        scores_second = scores[inds_second]   # 低置信度分数

        # 为高置信度检测框创建STrack对象
        if len(dets) > 0:
            '''创建检测轨迹对象'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        # 步骤 1: 将已跟踪的轨迹分为已确认和未确认两类
        unconfirmed = []         # 未确认的轨迹（只有一帧）
        tracked_stracks = []     # type: list[STrack]  # 已确认的活跃轨迹
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)      # 未激活的轨迹加入未确认列表
            else:
                tracked_stracks.append(track)  # 已激活的轨迹加入活跃列表

        # 步骤 2: 第一阶段关联 - 使用高置信度检测框
        # 合并活跃轨迹和丢失轨迹作为匹配候选
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # 使用卡尔曼滤波器预测当前帧中轨迹的位置
        STrack.multi_predict(strack_pool)
        
        # 计算轨迹和检测框之间的IoU距离
        dists = matching.iou_distance(strack_pool, detections)
        
        # 如果不是MOT20数据集，融合检测框分数到距离矩阵
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        
        # 线性分配算法进行匹配（匈牙利算法）
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
        
        # 处理成功匹配的轨迹
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # 更新已跟踪轨迹
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # 重新激活丢失的轨迹
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 步骤 3: 第二阶段关联 - 使用低置信度检测框
        # 为低置信度检测框创建STrack对象
        if len(dets_second) > 0:
            '''创建低置信度检测轨迹对象'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        
        # 获取第一阶段未匹配的已跟踪轨迹
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        # 计算未匹配轨迹和低置信度检测框的距离
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        
        # 进行线性分配，使用较低的匹配阈值
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        
        # 处理第二阶段的匹配结果
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 处理未匹配的轨迹 - 将其标记为丢失
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 步骤 4: 处理未确认轨迹（通常是只有一帧的轨迹）
        # 为第一阶段未匹配的检测框重新创建列表
        detections = [detections[i] for i in u_detection]
        
        # 计算未确认轨迹和未匹配检测框的距离
        dists = matching.iou_distance(unconfirmed, detections)
        
        # 融合分数到距离矩阵
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        
        # 进行线性分配，使用更高的阈值（0.7）
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        # 更新成功匹配的未确认轨迹
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        
        # 删除未匹配的未确认轨迹
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 步骤 5: 为新检测框初始化新轨迹
        for inew in u_detection:
            track = detections[inew]
            # 只有分数高于检测阈值的才创建新轨迹
            if track.score < self.det_thresh:
                continue
            # 使用卡尔曼滤波器激活新轨迹
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        # 步骤 6: 更新轨迹状态 - 删除超时的丢失轨迹
        for track in self.lost_stracks:
            # 如果轨迹丢失时间超过缓冲区大小，则删除
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 步骤 7: 更新全局轨迹列表
        # 只保留状态为"已跟踪"的轨迹
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 合并已跟踪轨迹和新激活轨迹
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        # 合并重新找到的轨迹
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # 从丢失列表中移除已跟踪的轨迹
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 添加新丢失的轨迹
        self.lost_stracks.extend(lost_stracks)
        # 从丢失列表中移除已删除的轨迹
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # 添加新删除的轨迹
        self.removed_stracks.extend(removed_stracks)
        # 移除重复的轨迹
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # 获取输出结果 - 返回所有已激活的轨迹
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks
    
    

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def btrack(tracker:BYTETracker,
           boxes:np.ndarray,
           scores:np.ndarray,
           ) -> tuple[list,
                      list,
                      list,
                      ]:
    """ByteTracker tracking interface

    Args:
        tracker (BYTETracker): _description_
        boxes (np.ndarray): xyxy格式的检测框，形状为(1, 4, n)
        scores (np.ndarray): _description_

    Returns:
        tuple[list, list, list]: _description_
    """
    output_results = np.concatenate((boxes[0].T, scores[0].reshape(-1, 1)), axis=1)
    online_targets = tracker.update(output_results)
    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        online_tlwhs.append(tlwh)
        online_ids.append(tid)
        online_scores.append(t.score)
    return online_tlwhs, online_ids, online_scores