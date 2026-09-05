"""One camera-frame estimate with bounded, transactional out-of-order fusion.

The short-horizon constant-velocity model has no invented table contact. Camera
measurements and their covariance stay in one frame even when table validity
changes. Scores are ranking scores, not calibrated probabilities.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Observation:
    t: float
    p: np.ndarray
    R: np.ndarray
    key: tuple
    source: str
    confidence: float


class TimestampedBallFilter:
    def __init__(self, lag_seconds=.20):
        self.lag_seconds = lag_seconds
        self.events = []
        self.states = []
        self.anchor = None
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.timestamp_s = None
        self.last_measurement_timestamp_s = None
        self.initialized = False
        self.last_confidence = 0.
        self.last_reason = 'uninitialized'
        self.counters = {}
        self._serial = 0

    def reset(self):
        self.__init__(self.lag_seconds)

    @staticmethod
    def propagate(state, t):
        x, P, old_t = state
        dt = float(t-old_t)
        if dt < -1.e-7:
            raise ValueError('Cannot propagate backwards; use a historical state')
        dt = max(0., dt)
        F = np.eye(6); F[:3, 3:] = np.eye(3)*dt
        # Continuous white acceleration, responsive to flight/impact innovation.
        # Spectral density is not per-frame acceleration variance: using 25
        # here would inject ~30x the previous velocity noise at 30 Hz.
        q = .75
        Q = np.block([[np.eye(3)*dt**3/3, np.eye(3)*dt**2/2],
                      [np.eye(3)*dt**2/2, np.eye(3)*dt]])*q
        return F@x, F@P@F.T+Q, float(t)

    @staticmethod
    def correct(state, event, check=True):
        if state is None:
            x = np.r_[event.p, np.zeros(3)]
            P = np.zeros((6, 6)); P[:3, :3] = event.R
            P[3:, 3:] = np.eye(3)*9.
            return x, P, event.t
        x, P, _ = TimestampedBallFilter.propagate(state, event.t)
        residual = event.p-x[:3]
        S = P[:3, :3]+event.R
        mahal = float(residual@np.linalg.solve(S, residual))
        if check and mahal > 16.3:
            return None
        K = np.linalg.solve(S, P[:3, :]).T
        A = np.eye(6); A[:, :3] -= K
        P = A@P@A.T+K@event.R@K.T
        return x+K@residual, (P+P.T)*.5, event.t

    def _reason(self, reason, accepted=False):
        self.last_reason = reason
        self.counters[reason] = self.counters.get(reason, 0)+1
        return accepted

    def update(self, p, t, confidence, covariance=None, key=None, source='unknown'):
        p = np.asarray(p, float)
        R = np.asarray(covariance if covariance is not None else np.eye(3)*.035**2, float)
        if p.shape != (3,) or R.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(R).all():
            return self._reason('invalid_measurement')
        R = (R+R.T)*.5
        if not np.isfinite(t) or np.linalg.eigvalsh(R).min() <= 0:
            return self._reason('invalid_covariance_or_timestamp')
        if key is None:
            self._serial += 1; key = ('anonymous', self._serial)
        if any(e.key == key for e in self.events):
            return self._reason('duplicate_measurement')
        if self.initialized and (t < self.timestamp_s-self.lag_seconds or
                                 (self.anchor is not None and t <= self.anchor[2])):
            return self._reason('outside_history')
        event = Observation(float(t), p.copy(), R.copy(), tuple(key), source, float(confidence))
        events = list(self.events)
        # Aligned depth and IR stereo around one acquisition are correlated.
        # Keep the lower-variance observation instead of counting both.
        family = 'rgb' if source.startswith('rgb') else 'ir' if source.startswith('ir') else source
        correlated = [e for e in events if abs(e.t-t) <= .012 and
                      {family, 'rgb' if e.source.startswith('rgb') else 'ir' if e.source.startswith('ir') else e.source} == {'rgb', 'ir'}]
        if correlated:
            if any(np.trace(e.R) <= np.trace(R) for e in correlated):
                return self._reason('correlated_measurement')
            replaced = {id(e) for e in correlated}
            events = [e for e in events if id(e) not in replaced]
        if any(abs(e.t-t) < 1.e-7 for e in events):
            return self._reason('duplicate_timestamp')
        events.append(event); events.sort(key=lambda e: e.t)
        state = self.anchor
        states = []
        for e in events:
            state = self.correct(state, e)
            if state is None:
                return self._reason('innovation_rejected')
            states.append(state)
        was_late = self.initialized and t < self.timestamp_s
        # Atomic commit: a rejected delayed observation leaves current state intact.
        while len(events) > 1 and events[0].t < events[-1].t-self.lag_seconds:
            self.anchor = states.pop(0); events.pop(0)
        self.events, self.states = events, states
        self.x, self.P, self.timestamp_s = [v.copy() if isinstance(v, np.ndarray) else v for v in states[-1]]
        self.last_measurement_timestamp_s = self.timestamp_s
        self.last_confidence = events[-1].confidence
        self.initialized = True
        return self._reason('late_measurement_fused' if was_late else 'measurement_fused', True)

    def predict_state(self, t):
        if not self.initialized:
            return None
        eligible = [s for s in self.states if s[2] <= t+1.e-7]
        state = eligible[-1] if eligible else self.anchor
        if state is None or t < state[2]-1.e-7:
            return None
        x, P, _ = self.propagate(state, t)
        return x, P

    def measurement_age(self, t):
        return float('inf') if not self.initialized else max(0., t-self.last_measurement_timestamp_s)

    def confidence(self, t):
        return float(self.last_confidence*np.exp(-self.measurement_age(t)/.18)) if self.initialized else 0.


def measurement_covariance(point, source, focal=434., baseline=.095, sigma=.035):
    """Depth uncertainty grows with Z^2/(f B); lateral precision is higher.

Conservative defaults include alignment/centroid bias. These are engineering
noise floors awaiting calibration on labelled motion, not accuracy guarantees.
"""
    p = np.asarray(point, float); z = max(.2, p[2])
    if source.startswith('ir'):
        depth_sigma = max(float(sigma), z*z/max(focal*baseline, .001)*.8)
        pixel_sigma = 1.2
        lateral_floor = .012  # Includes IR/RGB centre-definition uncertainty.
    else:
        depth_sigma = max(.012, float(sigma)*.6)
        pixel_sigma = .8
        lateral_floor = .004
    J = np.array([[z/focal, 0., p[0]/z], [0., z/focal, p[1]/z], [0., 0., 1.]])
    return J@np.diag([pixel_sigma**2, pixel_sigma**2, depth_sigma**2])@J.T+np.diag([lateral_floor**2,lateral_floor**2,.004**2])
