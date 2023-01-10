import numpy as np
import joblib
import json
import argparse

def get_camera_parameters(pred_cam, bbox):
    FOCAL_LENGTH = 5000.
    CROP_SIZE = 224

    bbox_cx, bbox_cy, bbox_w, bbox_h = bbox
    assert bbox_w == bbox_h

    bbox_size = bbox_w
    bbox_x = bbox_cx - bbox_w / 2.
    bbox_y = bbox_cy - bbox_h / 2.

    scale = bbox_size / CROP_SIZE

    cam_intrinsics = np.eye(3)
    cam_intrinsics[0, 0] = FOCAL_LENGTH * scale
    cam_intrinsics[1, 1] = FOCAL_LENGTH * scale
    cam_intrinsics[0, 2] = bbox_size / 2. + bbox_x
    cam_intrinsics[1, 2] = bbox_size / 2. + bbox_y

    cam_s, cam_tx, cam_ty = pred_cam
    trans = [cam_tx, cam_ty, 2*FOCAL_LENGTH/(CROP_SIZE*cam_s + 1e-9)]

    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, 3] = trans

    return cam_intrinsics, cam_extrinsics

parser = argparse.ArgumentParser()
parser.add_argument('--data_id', type=str,
                        help='input video path')
parser.add_argument('--output_path', type=str,
                        help='output video path')
args = parser.parse_args()

output = joblib.load('output/'+args.data_id+'/vibe_output.pkl')[1]
frames = output['pose'].shape[0]
print("Read File:{}".format('output/'+args.data_id+'/vibe_output.pkl'))
print("Output File:{}".format(args.output_path+'metadata.json'))
print("Frames:{}".format(frames))
data = {}
for i in range(frames):
    image_id = str(i+1).zfill(6)
    data[image_id] = {}
    data[image_id]["poses"] = output['pose'][i].tolist()
    data[image_id]["betas"] = output['betas'][i].tolist()
    cami, came = get_camera_parameters(output['pred_cam'][i], output['bboxes'][i])
    data[image_id]["cam_intrinsics"] = cami.tolist()
    data[image_id]["cam_extrinsics"] = came.tolist()

b = json.dumps(data)
f2 = open(args.output_path+'metadata.json', 'w')
f2.write(b)
f2.close()

