"""Single-image SPNv2 inference; run from the repository root."""
import argparse
import json
from pathlib import Path

import _init_paths
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from config import cfg, update_config
from dataset.transforms.build import build_transforms
from nets import build_spnv2
from utils.postprocess import solve_pose_from_heatmaps, rot_6d_to_matrix
from utils.utils import load_camera_intrinsics, load_tango_3d_keypoints


def offline_backbone(*args, **kwargs):
    """Build B3 without downloading initialization replaced by the checkpoint."""
    from nets.backbones.efficientnet import efficientnet_b3
    kwargs['pretrained'] = False
    return efficientnet_b3(*args, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg', default='experiments/demo_phi3_BN.yaml')
    p.add_argument('--image', default='/home/xuliheng/newProjects/datasets/speedplus/synthetic/images/img000014.jpg')
    p.add_argument('--output-dir', default='demo_outputs/img000014')
    p.add_argument('--device', default='auto')
    p.add_argument('opts', nargs=argparse.REMAINDER)
    args = p.parse_args()
    update_config(cfg, args)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    h, w = image.shape[:2]
    if [w, h] != list(cfg.DATASET.IMAGE_SIZE):
        raise ValueError('Use an original-resolution image matching DATASET.IMAGE_SIZE and the camera calibration')
    # Match tools/preprocess.py: grayscale, resize, then three equal channels.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, tuple(cfg.DATASET.INPUT_SIZE))
    tensor = build_transforms(cfg, is_train=False, load_labels=False)(image=cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR))['image'].unsqueeze(0)
    # The full checkpoint replaces ImageNet initialization; avoid network downloads.
    import nets.backbones.efficientdet as backbone
    backbone.efficientnet_b3 = offline_backbone
    model = build_spnv2(cfg)
    state = torch.load(cfg.TEST.MODEL_FILE, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    with torch.no_grad():
        outputs = dict(zip(cfg.TEST.HEAD, model(tensor, is_train=False, gpu=device)))
    heat = outputs['heatmap'][0].cpu().contiguous()
    confidence, index = heat.flatten(1).max(1)
    xy = np.column_stack(((index.numpy() % heat.shape[2]) / heat.shape[2] * w,
                          (index.numpy() // heat.shape[2]) / heat.shape[1] * h))
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    points = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)
    _, q, _, rejected = solve_pose_from_heatmaps(heat, [w, h], cfg.TEST.HEATMAP_THRESHOLD, camera, points)
    cls, boxes, rotation, translation = outputs['efficientpose']
    anchor = int(cls.reshape(-1).argmax())
    bbox = boxes[0, anchor].cpu().numpy().copy()
    bbox *= np.array([w / cfg.DATASET.INPUT_SIZE[0], h / cfg.DATASET.INPUT_SIZE[1]] * 2)
    bbox_image = image.copy()
    x1, y1, x2, y2 = np.rint(bbox).astype(int)
    cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    reg_R = rot_6d_to_matrix(rotation[0, anchor]).cpu().numpy()
    t = translation[0, anchor].cpu().numpy()
    # PnP returns object-to-camera rotation; regression uses the opposite convention.
    R = reg_R.T if rejected else Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()
    rvec = cv2.Rodrigues(R)[0]
    projection = image.copy()
    projected, _ = cv2.projectPoints(points, rvec, t, camera['cameraMatrix'], camera['distCoeffs'])
    keypoint_image = image.copy()
    for panel, coords in [(keypoint_image, xy), (projection, projected.reshape(-1, 2))]:
        for i, (x, y) in enumerate(coords):
            if np.isfinite([x, y]).all() and 0 <= x < w and 0 <= y < h:
                cv2.circle(panel, (int(x), int(y)), 7, (0, 255, 0), -1)
                cv2.putText(panel, str(i), (int(x)+8, int(y)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    prob = cv2.resize(outputs['segmentation'].sigmoid()[0, 0].cpu().numpy(), (w, h))
    mask = (prob > .5).astype(np.uint8) * 255
    overlay = image.copy()
    overlay[mask > 0] = (.55 * image[mask > 0] + .45 * np.array([0, 255, 0])).astype(np.uint8)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    panels = [('original', image), ('keypoints', keypoint_image), ('mask_overlay', overlay), ('projection', projection)]
    for name, panel in panels + [('mask', mask), ('bbox', bbox_image)]:
        if not cv2.imwrite(str(out / (name+'.png')), panel):
            raise IOError('Cannot write output image')
    thumbnails = []
    for name, panel in panels:
        thumb = cv2.resize(panel, (640, 400))
        cv2.putText(thumb, name, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        thumbnails.append(thumb)
    cv2.imwrite(str(out/'comparison.png'), np.vstack((np.hstack(thumbnails[:2]), np.hstack(thumbnails[2:]))))
    result = dict(image=str(Path(args.image).resolve()), checkpoint=cfg.TEST.MODEL_FILE,
                  keypoints=xy.tolist(), confidence=confidence.tolist(), pnp_rejected=bool(rejected),
                  rotation_object_to_camera=R.tolist(), translation_m=t.tolist(),
                  pose_source='efficientpose_fallback' if rejected else 'heatmap_rotation_efficientpose_translation',
                  projection_type='11_model_keypoints_not_CAD_mesh', device=str(device))
    result['bbox_xyxy_original_pixels'] = bbox.tolist()
    result['detection_score'] = float(cls.reshape(-1)[anchor].cpu())
    (out/'prediction.json').write_text(json.dumps(result, indent=2))
    print('DEMO_SUCCESS', out.resolve(), 'pnp_rejected=', rejected, flush=True)


if __name__ == '__main__':
    main()
