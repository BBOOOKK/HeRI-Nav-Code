"""Run one HeRI-Nav network on complete sensor windows and nominal positions."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import torch

def relative_heading(windows):
    x = np.asarray(windows)
    if x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (200, 14) or len(x) == 0:
        raise ValueError("Expected nonempty float32 (N,200,14) windows")
    if not np.isfinite(x).all() or np.any(x[:, :, 9] <= 0):
        raise ValueError("Invalid sensor window")
    if not np.allclose(x[:, :, 12]**2 + x[:, :, 13]**2, 1., atol=2e-6, rtol=0.):
        raise ValueError("Invalid sensor heading pair")
    result = x.copy()
    s, c = x[:, :, 12], x[:, :, 13]
    result[:, :, 12] = s*c[:, :1] - c*s[:, :1]
    result[:, :, 13] = c*c[:, :1] + s*s[:, :1]
    return result

def zyx_rotation(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.stack((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                     sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                     -sp, cp*sr, cp*cr), axis=-1).reshape(*np.shape(yaw), 3, 3)

def source_sensor_basis(x, center_roll_pitch):
    """Rotate all vector streams and attitudes; retain the initial yaw.

    The basis is constant within each completed window and determined by its
    first attitude. Physical estimator states and output transport frames are
    separate inputs and are not changed here. The center is fitted on source
    sensor orientations, with equal mission weights.
    """
    x = np.asarray(x)
    center = np.asarray(center_roll_pitch, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (200, 14) or not len(x):
        raise ValueError('Expected nonempty N x 200 x 14 sensor windows.')
    if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(x).all():
        raise ValueError('Finite windows and a two-element roll/pitch center are required.')
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError('Sensor windows must have a floating-point dtype.')
    raw = x.astype(np.float64)
    n = len(x)
    initial = zyx_rotation(np.zeros(n), raw[:, 0, 11], raw[:, 0, 10])
    target = zyx_rotation(np.zeros(n), np.full(n, center[1]), np.full(n, center[0]))
    change = target.swapaxes(-1, -2) @ initial
    value = raw.copy()
    for k in (0, 3, 6):
        value[..., k:k+3] = np.einsum('nij,ntj->nti', change, raw[..., k:k+3])
    yaw = np.arctan2(raw[..., 12], raw[..., 13])
    rotation = zyx_rotation(yaw, raw[..., 11], raw[..., 10]) @ change[:, None].swapaxes(-1, -2)
    horizontal = np.hypot(rotation[..., 0, 0], rotation[..., 1, 0])
    if np.any(horizontal < 1e-5):
        raise ValueError('The projected attitude is at an Euler pitch singularity.')
    yaw = np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])
    value[..., 10] = np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2])
    value[..., 11] = np.arctan2(-rotation[..., 2, 0], horizontal)
    value[..., 12], value[..., 13] = np.sin(yaw), np.cos(yaw)
    return value.astype(x.dtype)


def infer(network, input_file):
    with np.load(input_file, allow_pickle=False) as z:
        x = relative_heading(source_sensor_basis(z['x'], [-0.015313691364554474, -0.11279979749928867]))
        nominal = np.asarray(z['nominal'], dtype=float)
        yaw = np.asarray(z['yaw'], dtype=float)
        stamps = z['stamps'].copy()
    if nominal.shape != (len(x), 2) or yaw.shape != (len(x),):
        raise ValueError('Expected one nominal endpoint and start yaw per complete window')
    if not np.isfinite(nominal).all() or not np.isfinite(yaw).all():
        raise ValueError('Nonfinite physical input')
    with torch.inference_mode():
        mu = torch.cat([network(torch.from_numpy(x[i:i+64]))
                        for i in range(0, len(x), 64)]).numpy().astype(float)
    c, s = np.cos(yaw), np.sin(yaw)
    correction = np.column_stack((c*mu[:, 0]-s*mu[:, 1], s*mu[:, 0]+c*mu[:, 1]))
    return dict(mu=mu, position=nominal+np.cumsum(correction, axis=0), stamps=stamps)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['predict', 'validate'])
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    folder=Path(__file__).resolve().parent
    manifest=json.loads((folder/'manifest.json').read_text())
    groups=['inference_files'] + (['validation_files'] if args.command=='validate' else [])
    for group in groups:
        for name, expected in manifest[group].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Changed bundle file: '+name)
    torch.set_num_threads(2)
    network=torch.jit.load(str(folder/'model.pt'),map_location='cpu').eval()
    if args.command=='predict':
        if args.input is None: parser.error('--input is required for prediction')
        output=infer(network,args.input)
        with args.output.open('xb') as f: np.savez_compressed(f,**output)
        print(json.dumps(dict(status='complete',intervals=len(output['mu']),output=str(args.output))))
        return
    expected=json.loads((folder/'expected.json').read_text())
    maximum=0.; endpoints=0
    for case in manifest['cases']:
        result=infer(network,folder/case['input'])
        truth=np.asarray(expected[case['id']],float)
        if truth.shape!=result['position'].shape: raise ValueError('Endpoint count differs')
        difference=float(np.max(np.abs(truth-result['position'])))
        maximum=max(maximum,difference); endpoints+=len(truth)
    if maximum>1e-5: raise ValueError('Export differs from frozen predictions: '+str(maximum))
    report=dict(verified=True,model_name='HeRI-Nav',networks=1,seed=42,
                configuration_segments=len(manifest['cases']),endpoints=endpoints,
                maximum_position_difference_m=maximum,
                scope='Neural correction replay over frozen physical-estimator positions and headings',
                model_sha256=hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest())
    with args.output.open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps(report))

if __name__=='__main__': main()
