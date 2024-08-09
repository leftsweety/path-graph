import json

from requests import Session

from concentriq.basis import ThreadSafeRequestsSession


class ImageSets:
    def __init__(self, concentriq_endpoint: str, session: ThreadSafeRequestsSession):
        self.endpoint = f"{concentriq_endpoint}/imageSets"
        self.session = session

    def create_imageset(self, imageset: dict):
        return self.session.post(self.endpoint, json=imageset, verify=False)

    def get_image_sets(self):
        res = []
        page = 1
        while True:
            params = {
                "pagination": json.dumps({"rowsPerPage": 1000, "sortBy": "name", "descending": False, "page": page})
            }
            result = self.session.get(self.endpoint, params=params, verify=False)
            data = result["data"]["imageSets"]
            pagination = result["meta"]["pagination"]
            res.extend(data)
            if pagination["rowsPerPage"] * page < pagination["totalRows"]:
                page += 1
            else:
                break

        return res
