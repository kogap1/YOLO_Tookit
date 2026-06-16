from tracker.byte_tracker import BYTETracker, btrack, btrack_old
import numpy as np


if __name__ == "__main__":
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False
    tracker = BYTETracker(track_thresh, track_buffer, match_thresh, mot20)
    
    nob = 5
    boxes_old = np.random.rand(nob, 4)
    scores_old = np.random.rand(nob,)
    print("boxes: ", boxes_old.shape)
    print("scores: ", scores_old.shape)
    online_tlwhs, online_ids, online_scores = btrack_old(tracker, boxes_old, scores_old)

    boxes = np.random.rand(1, 4, nob)
    scores = np.random.rand(1, nob)
    print("boxes: ", boxes.shape)
    print("scores: ", scores.shape)
    online_tlwhs, online_ids, online_scores = btrack(tracker, boxes, scores)