import json

from concentriq.basis import ThreadSafeRequestsSession


class Annotations:
    def __init__(self, concentriq_endpoint: str, session: ThreadSafeRequestsSession):
        self.endpoint = f"{concentriq_endpoint}/annotations"
        self.session = session

    def get_annotations(self, filters):
        return self.session.get(self.endpoint,
                                params={"filters": json.dumps(filters)}, verify=False)

    def get_annotations_by_image_id(self, image_id):
        return self.get_annotations({"imageId": [image_id]})
