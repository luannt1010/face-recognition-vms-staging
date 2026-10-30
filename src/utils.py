import os
import cv2
from pathlib import Path
import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection() -> psycopg.Connection:
    # Docker dùng PGHOST=db và tham số riêng để mật khẩu không cần URL-encode.
    if os.getenv("PGHOST"):
        connection = psycopg.connect(
            host=os.environ["PGHOST"], port=os.getenv("PGPORT", "5432"),
            dbname=os.getenv("POSTGRES_DB", "face_recognition"),
            user=os.getenv("POSTGRES_USER", "face_app"),
            password=os.environ["POSTGRES_PASSWORD"], row_factory=dict_row,
        )
    elif DATABASE_URL:
        connection = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    else:
        raise RuntimeError(f"Not found DATABASE_URL in {ENV_PATH}")
    register_vector(connection)
    return connection


def check_database_connection() -> bool:
    try:
        with get_connection() as connection:
            result = connection.execute(
                """
                SELECT
                    current_user AS database_user,
                    current_database() AS database_name,
                    EXISTS (
                        SELECT 1
                        FROM pg_extension
                        WHERE extname = 'vector'
                    ) AS vector_enabled;
                """
            ).fetchone()

        print("Connect database successfully!")
        print("User:", result["database_user"])
        print("Database:", result["database_name"])
        print("pgvector:", result["vector_enabled"])
        return True

    except psycopg.Error as error:
        print("Connect database failed:")
        print(error)
        return False

def crop_face(frame: np.ndarray, bboxes):
    results = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        res = frame[y1:y2, x2:x1]
        results.append(res)
    return results

def draw_frame(tracked_result: dict, frame: np.ndarray, color: tuple | str):
    for result in tracked_result:
        idd = result["track_id"]
        bbox = result["bbox"]
        score = result["score"]
        label = f"{idd}: {round(score, 2)}"
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, pt1=[x1, y1], pt2=[x2, y2], color=color, thickness=2)
        cv2.putText(frame, text=label, org=[x1,y1-10], color=color, thickness=2, fontScale=0.6, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    return frame

def box_process(position, points, anchors):
    """Process the predicted boxes and landmarks.
    """
    IMG_SIZE = (640, 640)
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:,:2,:,:]*2 - 0.5
    box_wh = pow(position[:,2:4,:,:]*2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    points[:,0:2,:,:] = points[:,0:2,:,:] * anchors + grid * stride
    points[:,2:4,:,:] = points[:,2:4,:,:] * anchors + grid * stride
    points[:,4:6,:,:] = points[:,4:6,:,:] * anchors + grid * stride
    points[:,6:8,:,:] = points[:,6:8,:,:] * anchors + grid * stride
    points[:,8:10,:,:] = points[:,8:10,:,:] * anchors + grid * stride
    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2  # bottom right y
    return xyxy, points


def nms_boxes(boxes, scores, NMS_THRESH):
    """Suppress non-maximal boxes using cv2.dnn.NMSBoxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    # Convert from (x1, y1, x2, y2) to (x, y, w, h) format
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    boxes_xywh = np.column_stack([x1, y1, w, h])
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=NMS_THRESH
    )
    if len(indices) > 0:
        keep = indices.flatten()
    else:
        keep = np.array([], dtype=np.int32)
    return keep

def filter_boxes(boxes, box_confidences, box_class_probs, points, OBJ_THRESH):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    points = points[_class_pos]

    return boxes, classes, scores, points

def yolov5_facepostprocess(input_data,OBJ_THRESH,NMS_THRESH):
    """Post-process the outputs from the YOLOv5 model to decode bounding boxes and
    landmark points, and scale them back to the original image size.

    Args:
        input_data (list): The predicted outputs from the YOLOv5 model.
        OBJ_THRESH (float): The object confidence threshold for filtering out weak detections.
        NMS_THRESH (float): The non-maximum suppression threshold.

    Returns:
        dets (numpy.ndarray): The decoded and scaled bounding boxes of detected faces.
        lms (numpy.ndarray): The landmarks associated with the detected faces, scaled to the original image size.
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    anchors = np.array([[[4.0, 5.0], [8.0, 10.0], [13.0, 16.0]], [[23.0, 29.0], [43.0, 55.0], [73.0, 105.0]], [[146.0, 217.0], [231.0, 300.0], [335.0, 433.0]]])
    boxes, scores, classes_conf, points = [], [], [], []
    # 1*255*h*w -> 3*85*h*w
    input_data = [_in.reshape([len(anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
        input_data[i][:,:5,:,:] = sigmoid(input_data[i][:,:5,:,:])
        bbs, pps = box_process(input_data[i][:,:4,:,:], input_data[i][:,5:15,:,:], anchors[i])
        boxes.append(bbs)
        points.append(pps)
        scores.append(input_data[i][:,4:5,:,:])
        input_data[i][:,15:,:,:] = sigmoid(input_data[i][:,15:,:,:])
        classes_conf.append(input_data[i][:,15:,:])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    points = [sp_flatten(_v) for _v in points]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    points = np.concatenate(points)

    # filter according to threshold
    boxes, classes, scores, points = filter_boxes(boxes, scores, classes_conf, points, OBJ_THRESH)
    # nms
    nboxes, nclasses, nscores, npoints = [], [], [], []

    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        p = points[inds]
        keep = nms_boxes(b, s,NMS_THRESH)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
            npoints.append(p[keep])

    if not nclasses and not nscores:
        return None, None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    points = np.concatenate(npoints)
    return boxes, classes, scores, points

def l2_normalize(embedding, eps=1e-6):
    return np.divide(embedding, np.maximum(np.linalg.norm(embedding, axis=-1, keepdims=True), eps))



if __name__ == "__main__":
    check_database_connection()
