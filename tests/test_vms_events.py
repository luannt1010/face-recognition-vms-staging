import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.core_ai.event_processor import EventProcessor
from src.vms_service.vms_client import VMSClient

POLYGON = {"id": "p1", "ai_modules": "FACE", "points": [
    {"x": .3, "y": .3}, {"x": .7, "y": .3},
    {"x": .7, "y": .7}, {"x": .3, "y": .7}]}
LINE = {"id": "l1", "name": "OUT", "shape_type": "line_crossing",
        "line": {"points": [{"x": .2, "y": .5}, {"x": .8, "y": .5}]},
        "direction_point": {"x": .5, "y": .8}}


def track(x, y, track_id=1, real=True):
    return {"track_id": track_id, "bbox": [x*100-2, y*100-2, x*100+2, y*100+2],
            "score": .9, "is_detection": real}


class ProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = EventProcessor(cooldown=0)

    def step(self, x, y, now, zones=None, real=True, camera="CAM"):
        return self.processor.process(camera, [track(x, y, real=real)], (100, 100, 3),
                                      [POLYGON] if zones is None else zones, now=now)

    def test_polygon_enter_stay_exit(self):
        self.assertEqual(self.step(.1, .5, 0), [])
        self.assertEqual(self.step(.5, .5, 1)[0]["direction"], "IN")
        self.assertEqual(self.step(.6, .5, 2), [])
        self.assertEqual(self.step(.9, .5, 3)[0]["direction"], "OUT")

    def test_new_track_inside_does_not_create_entry(self):
        self.assertEqual(self.step(.5, .5, 0), [])

    def test_line_direction_uses_point_and_finite_segment(self):
        self.step(.5, .2, 0, [LINE])
        self.assertEqual(self.step(.5, .8, 1, [LINE])[0]["direction"], "IN")
        self.assertEqual(self.step(.5, .2, 2, [LINE])[0]["direction"], "OUT")
        self.step(.95, .2, 3, [LINE])
        self.assertEqual(self.step(.95, .8, 4, [LINE]), [])

    def test_lines_topic_point_format(self):
        line = dict(LINE)
        line["point"] = line.pop("direction_point")
        self.step(.5, .2, 0, [line])
        self.assertEqual(self.step(.5, .8, 1, [line])[0]["direction"], "IN")

    def test_boundary_and_prediction_do_not_trigger(self):
        self.step(.1, .5, 0)
        self.assertEqual(self.step(.301, .5, 1), [])
        self.assertEqual(self.step(.5, .5, 1.1, real=False), [])
        self.assertEqual(self.step(.5, .5, 1.2)[0]["direction"], "IN")

    def test_expiry_config_change_reset_and_camera_isolation(self):
        self.step(.1, .5, 0)
        self.assertEqual(self.step(.5, .5, 3), [])
        self.assertEqual(self.step(.1, .5, 3, camera="OTHER"), [])
        self.assertEqual(self.step(.1, .5, 4, []), [])
        self.assertEqual(self.step(.5, .5, 5), [])
        self.processor.reset("CAM")
        self.assertEqual(self.step(.1, .5, 6), [])

    def test_cooldown_does_not_delay_old_event(self):
        self.processor.cooldown = 2
        self.step(.1, .5, 0)
        self.assertEqual(self.step(.5, .5, .1)[0]["direction"], "IN")
        self.assertEqual(self.step(.1, .5, .2), [])
        self.assertEqual(self.step(.1, .5, 2), [])


class ClientTests(unittest.TestCase):
    def setUp(self):
        with patch("src.vms_service.vms_client.mqtt.Client") as factory:
            self.vms = VMSClient("example.invalid", "user", "password", zone_id=51)
        self.mqtt = factory.return_value
        self.message(self.vms.camera_topic, {"cameras": [
            {"id": "uuid", "code": "cổng/01", "ai_modules": ["FACE"]},
            {"id": "other", "code": "OFF", "ai_modules": []}]})

    def message(self, topic, data):
        payload = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.vms._on_message(None, None, SimpleNamespace(topic=topic, payload=payload))

    def test_config_update_empty_and_disabled(self):
        topic = "smart_vms/cameras/cổng_01/ai_zones/FACE"
        self.message(topic, {"ai_zones": [POLYGON, LINE]})
        self.assertEqual(len(self.vms.get_cameras()), 1)
        self.assertEqual(self.vms.get_zones("cổng/01"), [POLYGON, LINE])
        self.message(topic, b"")
        self.assertEqual(self.vms.get_zones("cổng/01"), [])
        self.message(self.vms.camera_topic, {"cameras": []})
        self.assertFalse(self.vms.publish_face("cổng/01", {}, {}))

    def test_empty_scope_and_disabled_modules_have_distinct_diagnostics(self):
        with self.assertLogs('src.vms_service.vms_client', level='INFO') as logs:
            self.message(self.vms.camera_topic, {'cameras': []})
        self.assertIn('received=0', ' '.join(logs.output))
        self.assertIn('zone_id=51', ' '.join(logs.output))
        with self.assertLogs('src.vms_service.vms_client', level='INFO') as logs:
            self.message(self.vms.camera_topic, {'cameras': [
                {'code': 'OFF', 'ai_modules': ['ZONE'], 'rtsp_url': 'rtsp://hidden-secret'}]})
        output = ' '.join(logs.output)
        self.assertIn('received=1', output)
        self.assertIn('OFF', output)
        self.assertIn('ZONE', output)
        self.assertNotIn('hidden-secret', output)
        self.assertEqual(self.vms.get_cameras(), [])

    def test_retained_zones_alone_do_not_enable_out_of_scope_camera(self):
        self.message(self.vms.camera_topic, {'cameras': []})
        with self.assertLogs('src.vms_service.vms_client', level='INFO') as logs:
            self.message('smart_vms/cameras/outside/ai_zones/FACE', {'ai_zones': [POLYGON]})
        self.assertIn('camera_selected=False', ' '.join(logs.output))
        self.assertEqual(self.vms.get_cameras(), [])
        self.assertEqual(self.vms.get_zones('outside'), [])

    def test_reconnect_subscribes_and_requests_again(self):
        for _ in range(2):
            self.vms._on_connect(self.mqtt, None, None, 0, None)
        self.assertEqual(self.mqtt.subscribe.call_count, 4)
        topic, body = self.mqtt.publish.call_args.args
        self.assertEqual(topic, "smart_vms/cameras/getdevice")
        self.assertEqual(json.loads(body), {"zone_id": 51, "organization_id": 103})
        self.vms._on_disconnect(None, None, None, 1, None)
        self.assertEqual(self.vms.get_cameras(), [])

    def test_face_payload_and_stranger(self):
        self.vms.connected.set()
        self.mqtt.publish.return_value.rc = 0
        self.mqtt.publish.return_value.is_published.return_value = True
        event = {"zone_id": "p1", "direction": "OUT"}
        for matched in (True, False):
            identity = {"matched": matched, "user_code": "EMP01", "score_match": .8}
            self.assertTrue(self.vms.publish_face("cổng/01", event, identity, "face/test.jpg"))
            args, kwargs = self.mqtt.publish.call_args
            body = json.loads(args[1])
            self.assertEqual(args[0], "smart_vms/ai_events/FACE")
            self.assertEqual(kwargs, {"qos": 1, "retain": False})
            self.assertEqual(body["direction"], "OUT")
            self.assertTrue(body["event_time"].endswith("+07:00"))
            self.assertEqual("user_code" in body, matched)

    def test_bbox_normalization_qos_and_real_detections(self):
        self.vms.connected.set()
        self.mqtt.publish.return_value.rc = 0
        self.vms.publish_bbox("cổng/01", [track(.5, .5), track(.1, .1, real=False)], (100, 100))
        args, kwargs = self.mqtt.publish.call_args
        self.assertEqual(args[0], "smart_vms/ai/bbox/cổng_01")
        self.assertEqual(kwargs, {"qos": 0, "retain": False})
        self.assertEqual(json.loads(args[1])["detections"][0]["bbox"], [.48, .48, .52, .52])
        self.assertEqual(len(json.loads(args[1])["detections"]), 1)
        self.assertEqual(json.loads(args[1])["detections"][0]["color"], "#ef4444")
        self.vms.publish_bbox("cổng/01", [track(.5, .5)], (100, 100),
                              identities={1: {"matched": True}})
        body = json.loads(self.mqtt.publish.call_args.args[1])
        self.assertEqual(body["detections"][0]["color"], "#22c55e")
        self.assertEqual(body['detections'][0]['class'], 'face')
        self.vms.publish_bbox('cổng/01', [track(.5, .5)], (100, 100), detection_class='person')
        body = json.loads(self.mqtt.publish.call_args.args[1])
        self.assertEqual(body['detections'][0]['class'], 'person')
        self.assertEqual(body['ai_modules'], 'FACE')

    def test_publish_failure_and_snapshot_object_key(self):
        self.assertFalse(self.vms._publish("topic", {}, 1))
        self.vms.connected.set()
        self.mqtt.publish.return_value.rc = 4
        self.assertFalse(self.vms._publish("topic", {}, 1))
        with patch("requests.post") as post:
            post.return_value.json.return_value = {"success": True, "object_key": "face/a.jpg"}
            self.assertEqual(self.vms.upload_snapshot("cổng/01", b"jpeg", "https://example.invalid"),
                             "face/a.jpg")
            self.assertEqual(post.call_args.kwargs["data"]["category"], "face")


if __name__ == "__main__":
    unittest.main()
