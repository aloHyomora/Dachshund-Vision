from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque
import numpy as np
import time

# Mediapipe 인덱스
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
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
    def compute(self, norm_pts: np.ndarray, world_pts: Optional[np.ndarray] = None) -> Dict:
        # 공통 유틸
        def angle_cos(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            # ∠ABC의 cos값
            v1 = a - b
            v2 = c - b
            n1 = np.linalg.norm(v1) + 1e-8
            n2 = np.linalg.norm(v2) + 1e-8
            return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))

        def is_extended_by_angle(mcp, pip, dip, tip, wrist, cos_thresh=-0.3, len_ratio=1.10) -> bool:
            # PIP 각도가 곧게 펴질수록 cos≈-1. 충분히 곧고 길이도 충분하면 펴짐으로.
            c = angle_cos(mcp, pip, tip)
            tip_len = np.linalg.norm(tip - wrist)
            mcp_len = np.linalg.norm(mcp - wrist) + 1e-8            
            return (c <= cos_thresh) and (tip_len / mcp_len >= len_ratio)

        wrist = norm_pts[WRIST, :2]
        # tip 사이 거리
        pinch_dist = float(np.linalg.norm(norm_pts[THUMB_TIP, :2] - norm_pts[INDEX_TIP, :2]))
        pinch_world_m = None
        if world_pts is not None and world_pts.shape[1] >= 3:
            a = world_pts[THUMB_TIP, :3]
            b = world_pts[INDEX_TIP, :3]
            pinch_world_m = float(np.linalg.norm(a - b))

        # 각 손가락 펴짐 판정
        # 엄지는 CMC–MCP–IP–TIP을 써서 각도/길이 판정이 더 안정적
        thumb_ext = is_extended_by_angle(
            norm_pts[THUMB_CMC, :2], norm_pts[THUMB_MCP, :2], norm_pts[THUMB_IP, :2], norm_pts[THUMB_TIP, :2],
            wrist, cos_thresh=-0.35, len_ratio=1.10
        )
        index_ext = is_extended_by_angle(
            norm_pts[INDEX_MCP, :2], norm_pts[INDEX_PIP, :2], norm_pts[INDEX_DIP, :2], norm_pts[INDEX_TIP, :2],
            wrist, cos_thresh=-0.3, len_ratio=1.10
        )
        middle_ext = is_extended_by_angle(
            norm_pts[MIDDLE_MCP, :2], norm_pts[MIDDLE_PIP, :2], norm_pts[MIDDLE_DIP, :2], norm_pts[MIDDLE_TIP, :2],
            wrist, cos_thresh=-0.3, len_ratio=1.10
        )
        ring_ext = is_extended_by_angle(
            norm_pts[RING_MCP, :2], norm_pts[RING_PIP, :2], norm_pts[RING_DIP, :2], norm_pts[RING_TIP, :2],
            wrist, cos_thresh=-0.3, len_ratio=1.08
        )
        pinky_ext = is_extended_by_angle(
            norm_pts[PINKY_MCP, :2], norm_pts[PINKY_PIP, :2], norm_pts[PINKY_DIP, :2], norm_pts[PINKY_TIP, :2],
            wrist, cos_thresh=-0.25, len_ratio=1.06
        )
        print(f"Debug: {thumb_ext}, {index_ext}, {middle_ext}, {ring_ext}, {pinky_ext}")
        fingers = {
            "thumb": thumb_ext,
            "index": index_ext,
            "middle": middle_ext,
            "ring": ring_ext,
            "pinky": pinky_ext,
        }
        return {
            "pinch_dist": pinch_dist,
            "pinch_world_m": pinch_world_m,
            "fingers": fingers,
        }

class StaticGestureRecognizer:
    def __init__(self,
                 pinch_norm_thresh: float = 0.15,
                 pinch_world_thresh_m: float = 0.03,
                 use_world: bool = True,
                 # POINT는 엄지-검지 거리가 이 값보다 멀어야 성립
                 point_min_pinch_norm: float = 0.20,
                 point_min_pinch_world_m: float = 0.05):
        self.pinch_norm_thresh = pinch_norm_thresh
        self.pinch_world_thresh_m = pinch_world_thresh_m
        self.use_world = use_world
        self.point_min_pinch_norm = point_min_pinch_norm
        self.point_min_pinch_world_m = point_min_pinch_world_m

    def classify(self, feats: Dict) -> str:
        f = feats["fingers"]
        pinch_world = feats.get("pinch_world_m")
        pinch_dist = feats["pinch_dist"]

        # 1) PINCH: 엄지+검지 펴짐 AND tip 거리 근접
        near = (self.use_world and pinch_world is not None and pinch_world < self.pinch_world_thresh_m) or \
               (pinch_world is None and pinch_dist < self.pinch_norm_thresh)
        if f["thumb"] and f["index"] and near:
            return "PINCH"

        # 2) OPEN: 전부 펴짐
        if all(f.values()):
            return "OPEN"

        # 3) FIST: 전부 접힘
        if not any(f.values()):
            return "FIST"

        # 4) POINT:
        # - 검지: 펴짐
        # - 중지/약지/새끼: 접힘
        # - 엄지-검지 거리가 충분히 멀다(엄지 펴짐 여부는 무시)
        far = (self.use_world and pinch_world is not None and pinch_world > self.point_min_pinch_world_m) or \
              (pinch_world is None and pinch_dist > self.point_min_pinch_norm)
        if f["index"] and not any([f["middle"], f["ring"], f["pinky"]]) and far:
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

    def update(self,
               hands_xy: List[np.ndarray],
               labels: List[str],
               ts_ms: int,
               hands_world: Optional[List[np.ndarray]] = None):
        tracks = self.tracker.assign(hands_xy, labels, ts_ms)
        overlay_labels: List[str] = []
        events: List[Dict] = []
        finger_states: List[Dict[str, bool]] = []

         # 입력 순서에 맞춰 라벨 생성
        for i, tr in enumerate(tracks):
            world_pts = hands_world[i] if hands_world is not None and i < len(hands_world) else None
            feats = self.extractor.compute(tr.last_norm_pts, world_pts)
            raw_label = self.static_recog.classify(feats)
            stable = self.smoother.update(tr.id, raw_label)
            finger_states.append(feats["fingers"])
            if stable is None:
                show = f"{tr.handedness}"
            else:
                show = f"{tr.handedness}:{stable}"
            overlay_labels.append(show)
            events.extend(self.fsm.update(tr, stable, ts_ms))

        return {"overlay_labels": overlay_labels, "events": events, "fingers": finger_states}