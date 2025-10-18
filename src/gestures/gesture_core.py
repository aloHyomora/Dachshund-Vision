from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque
import numpy as np
import time

# Mediapipe 인덱스
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17
PINKY_TIP = 20

@dataclass
class HandTrack:
    id: int
    handedness: str
    last_pts: np.ndarray  # image coords
    last_norm_pts: np.ndarray  # wrist/scale 정규화 좌표
    last_ts: int
    norm_wrist_hist: deque  # (ts, wrist_xy_normalized)

class HandTracker:
    def __init__(self, max_age_ms: int = 600, match_thresh_px: float = 120.0):
        self.tracks: Dict[int, HandTrack] = {}
        self.next_id = 0
        self.max_age_ms = max_age_ms
        self.match_thresh_px = match_thresh_px

    def _norm(self, pts: np.ndarray) -> np.ndarray:
        # wrist 기준 이동 + scale 정규화 (index_mcp–pinky_mcp)
        wrist = pts[WRIST, :2]
        span = np.linalg.norm(pts[INDEX_MCP, :2] - pts[PINKY_MCP, :2])
        if span < 1e-6: span = 1.0
        return (pts[:, :2] - wrist) / span

    def assign(self, hands_xy: List[np.ndarray], labels: List[str], ts_ms: int) -> List[HandTrack]:
        # 현재 프레임 손목 좌표
        curr = []
        for i, pts in enumerate(hands_xy):
            wrist_xy = pts[WRIST, :2]
            curr.append((i, wrist_xy, pts))

        # 기존 트랙과 그리디 매칭
        used = set()
        assignments: List[Tuple[int, int]] = []  # (track_id, curr_index)
        for tid, tr in list(self.tracks.items()):
            best = None
            best_d = 1e9
            for idx, wrist_xy, pts in curr:
                if idx in used: continue
                d = np.linalg.norm(tr.last_pts[WRIST, :2] - wrist_xy)
                if d < best_d:
                    best_d = d
                    best = (idx, pts)
            if best and best_d < self.match_thresh_px:
                idx, pts = best
                used.add(idx)
                assignments.append((tid, idx))
                norm = self._norm(pts)
                tr.last_pts = pts
                tr.last_norm_pts = norm
                tr.last_ts = ts_ms
                tr.norm_wrist_hist.append((ts_ms, norm[WRIST].copy()))
                while tr.norm_wrist_hist and ts_ms - tr.norm_wrist_hist[0][0] > self.max_age_ms:
                    tr.norm_wrist_hist.popleft()

        # 새 트랙 생성
        for idx, wrist_xy, pts in curr:
            if idx in used: continue
            norm = self._norm(pts)
            handedness = labels[idx] if idx < len(labels) else "Hand"
            tr = HandTrack(
                id=self.next_id, handedness=handedness,
                last_pts=pts, last_norm_pts=norm,
                last_ts=ts_ms, norm_wrist_hist=deque(maxlen=60)
            )
            tr.norm_wrist_hist.append((ts_ms, norm[WRIST].copy()))
            self.tracks[tr.id] = tr
            assignments.append((tr.id, idx))
            self.next_id += 1

        # 오래된 트랙 제거
        for tid, tr in list(self.tracks.items()):
            if ts_ms - tr.last_ts > self.max_age_ms:
                del self.tracks[tid]

        # 현재 프레임 순서대로 반환
        out: List[HandTrack] = []
        for tid, idx in sorted(assignments, key=lambda x: x[1]):
            out.append(self.tracks[tid])
        return out

class FeatureExtractor:
    def compute(self, norm_pts: np.ndarray) -> Dict:
        # 간단 특징들
        pinch_dist = float(np.linalg.norm(norm_pts[THUMB_TIP] - norm_pts[INDEX_TIP]))
        def extended(tip_idx: int, mcp_idx: int) -> bool:
            tip_d = np.linalg.norm(norm_pts[tip_idx])
            mcp_d = np.linalg.norm(norm_pts[mcp_idx]) + 1e-6
            return tip_d / mcp_d > 1.25  # 임계값은 튜닝 포인트
        fingers = {
            "thumb": extended(THUMB_TIP, INDEX_MCP),  # 근사
            "index": extended(INDEX_TIP, INDEX_MCP),
            "middle": extended(12, MIDDLE_MCP),
            "ring": extended(16, 13),
            "pinky": extended(PINKY_TIP, PINKY_MCP),
        }
        return {
            "pinch_dist": pinch_dist,
            "fingers": fingers,
        }

class StaticGestureRecognizer:
    def __init__(self,
                 pinch_thresh: float = 0.15,
                 fist_max_extended: int = 1,
                 open_min_extended: int = 4):
        self.pinch_thresh = pinch_thresh
        self.fist_max_extended = fist_max_extended
        self.open_min_extended = open_min_extended

    def classify(self, feats: Dict) -> str:
        fingers = feats["fingers"]
        extended_cnt = sum(fingers.values())
        print(feats["pinch_dist"])
        if feats["pinch_dist"] < self.pinch_thresh:
            return "PINCH"
        if extended_cnt >= self.open_min_extended:
            return "OPEN"
        if extended_cnt <= self.fist_max_extended:
            return "FIST"
        if fingers["index"] and not fingers["middle"]:
            return "POINT"
        return "UNKNOWN"

class TemporalSmoother:
    def __init__(self, window: int = 5, min_count: int = 3):
        self.window = window
        self.min_count = min_count
        self.hist: Dict[int, deque] = {}

    def update(self, hand_id: int, label: str) -> Optional[str]:
        q = self.hist.setdefault(hand_id, deque(maxlen=self.window))
        q.append(label)
        # 다수결
        vals, cnts = np.unique(list(q), return_counts=True)
        best = vals[int(np.argmax(cnts))]
        if int(np.max(cnts)) >= self.min_count:
            return best
        return None

class SequenceFSM:
    # 예시: PINCH 250ms 유지 후 500ms 내 오른쪽 스와이프
    def __init__(self, hold_ms: int = 250, swipe_window_ms: int = 500, swipe_dx_thresh: float = 0.6):
        self.hold_ms = hold_ms
        self.swipe_window_ms = swipe_window_ms
        self.swipe_dx_thresh = swipe_dx_thresh
        self.state: Dict[int, str] = {}   # hand_id -> state
        self.state_ts: Dict[int, int] = {}
        self.pinch_start_ts: Dict[int, int] = {}

    def update(self, tr: HandTrack, stable_label: Optional[str], ts_ms: int) -> List[Dict]:
        events = []
        sid = tr.id
        st = self.state.get(sid, "IDLE")

        if st == "IDLE":
            if stable_label == "PINCH":
                self.state[sid] = "PINCH_HOLD"
                self.pinch_start_ts[sid] = ts_ms

        elif st == "PINCH_HOLD":
            if stable_label != "PINCH":
                # 핀치 해제. 홀드 시간 만족했는지 확인
                hold_ok = ts_ms - self.pinch_start_ts.get(sid, ts_ms) >= self.hold_ms
                self.state[sid] = "LOOK_FOR_SWIPE"
                self.state_ts[sid] = ts_ms
                if not hold_ok:
                    # 조건 미달이면 즉시 리셋
                    self.state[sid] = "IDLE"
            # 계속 핀치 상태면 대기

        elif st == "LOOK_FOR_SWIPE":
            # 최근 swipe_window 내 손목 이동량 평가
            t0 = ts_ms - self.swipe_window_ms
            hist = [p for (t, p) in tr.norm_wrist_hist if t >= t0]
            if len(hist) >= 2:
                dx = hist[-1][0] - hist[0][0]
                if dx > self.swipe_dx_thresh:
                    events.append({"type": "PINCH_THEN_SWIPE_RIGHT", "hand_id": sid, "ts": ts_ms})
                    self.state[sid] = "IDLE"
            # 타임아웃
            if ts_ms - self.state_ts.get(sid, ts_ms) > self.swipe_window_ms:
                self.state[sid] = "IDLE"

        return events

class GesturePipeline:
    def __init__(self):
        self.tracker = HandTracker()
        self.extractor = FeatureExtractor()
        self.static_recog = StaticGestureRecognizer()
        self.smoother = TemporalSmoother()
        self.fsm = SequenceFSM()

    def update(self, hands_xy: List[np.ndarray], labels: List[str], ts_ms: int):
        tracks = self.tracker.assign(hands_xy, labels, ts_ms)
        overlay_labels: List[str] = []
        events: List[Dict] = []

        # 입력 순서에 맞춰 라벨 생성
        for tr in tracks:
            feats = self.extractor.compute(tr.last_norm_pts)
            raw_label = self.static_recog.classify(feats)
            stable = self.smoother.update(tr.id, raw_label)
            if stable is None:
                show = f"{tr.handedness}"
            else:
                show = f"{tr.handedness}:{stable}"
            overlay_labels.append(show)
            events.extend(self.fsm.update(tr, stable, ts_ms))

        return {"overlay_labels": overlay_labels, "events": events}