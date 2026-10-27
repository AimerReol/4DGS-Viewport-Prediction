import numpy as np

def _unit_quaternion(value):
    q = np.asarray(value, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError('Expected finite xyzw quaternions')
    length = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(length < 1e-12):
        raise ValueError('Zero quaternion is not a rotation')
    return q / length

def _forward(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)], axis=-1)

def _angle_between(a, b):

    sine = np.linalg.norm(np.cross(a, b), axis=-1)
    cosine = np.sum(a * b, axis=-1)
    return np.degrees(np.arctan2(sine, cosine))

def pose_errors(pred_position, pred_quaternion, true_position, true_quaternion):
    pp, tp = np.asarray(pred_position, np.float64), np.asarray(true_position, np.float64)
    pq, tq = _unit_quaternion(pred_quaternion), _unit_quaternion(true_quaternion)
    if pp.shape != tp.shape or pp.ndim < 1 or pp.shape[-1] != 3:
        raise ValueError('Position predictions and targets must have matching [...,3] shapes')
    if pq.shape != tq.shape or pq.shape[:-1] != pp.shape[:-1]:
        raise ValueError('Quaternion and position batch dimensions must match')
    if not np.isfinite(pp).all() or not np.isfinite(tp).all():
        raise ValueError('Positions must be finite')

    a, b = -tq[..., :3], pq[..., :3]
    aw, bw = tq[..., 3:4], pq[..., 3:4]
    vector = aw * b + bw * a + np.cross(a, b)
    scalar = (aw * bw)[..., 0] - np.sum(a * b, axis=-1)
    orientation = np.degrees(2 * np.arctan2(np.linalg.norm(vector, axis=-1), np.abs(scalar)))
    return dict(position_error_native_units=np.linalg.norm(pp - tp, axis=-1),
                orientation_geodesic_deg=orientation,
                viewing_direction_error_deg=_angle_between(_forward(pq), _forward(tq)))
