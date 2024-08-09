import json
from typing import Literal, Union

from concentriq.basis import ThreadSafeRequestsSession


class Images:
    def __init__(self, concentriq_endpoint: str, session: ThreadSafeRequestsSession):
        self.endpoint = f"{concentriq_endpoint}/images"
        self.session = session

    def create_image(self, image):
        return self.session.post(self.endpoint, json=image, verify=False)

    def get_image(self, image_id):
        return self.session.get(f"{self.endpoint}/{image_id}", verify=False)

    def update_image(self, image_id, image: dict):
        return self.session.patch(f"{self.endpoint}/{image_id}", json=image, verify=False)

    def delete_image(self, image_id):
        return self.session.delete(f"{self.endpoint}/{image_id}", verify=False)

    def copy_images_to(self, source_image_ids, image_set_id=None, folder_id=None, clear_metadata: bool = False):
        """
        Copy the source image
        :param source_image_ids:
        :param image_set_id:
        :param folder_id:
        :param clear_metadata:
        :return: the
        """
        if image_set_id is None and folder_id is None:
            raise Exception(f"image_set_id and folder_id cannot both be None.")

        if folder_id is None:
            response = self.session.post(f"{self.endpoint}/batch",
                                         json={
                                             "cloneSourceIds": source_image_ids,
                                             "imageSetId": image_set_id,
                                             "clearMetadata": clear_metadata
                                         },
                                         verify=False)
        else:
            response = self.session.post(f"{self.endpoint}/batch",
                                         json={
                                             "cloneSourceIds": source_image_ids,
                                             "folderParentId": folder_id,
                                             "clearMetadata": clear_metadata
                                         },
                                         verify=False)

        return [d["image_id"] for d in response["data"][0]]

    def duplicate_image(self, source_image_id, name: str):
        """
        Duplicate the source image and return the created image_id
        :param source_image_id:
        :param name: name of the duplicated image
        :return:
        """

        response = self.session.post(f"{self.endpoint}",
                                     json={
                                         "cloneSourceId": source_image_id,
                                         "name": name,
                                     },
                                     verify=False)

        return response["data"]["id"]

    def get_images(self, filters: dict):
        return self.session.get(self.endpoint, params={"filters": json.dumps(filters)}, verify=False)

    def get_images_by_imageset_id(self, image_set_id):
        return self.get_images({"imageSetId": [image_set_id]})

    def extract_image_patch(self, image_id,
                            top_left_x: int, top_left_y: int, bottom_right_x: int, bottom_right_y: int,
                            width: int, image_type: Union[Literal["jpg"], Literal["tif"]]) -> bytes:
        """
        Use the IIIF image server of Concentriq to extract a patch from the image.
        :param image_id: the Concentriq image id of the target image.
        :param top_left_x: the x coordinate of the top left point
        :param top_left_y: the y coordinate of the top left point
        :param bottom_right_x: the x coordinate of the bottom right point
        :param bottom_right_y: the y coordinate of the bottom right point
        :param width: the width in pixels of the extracted image. This defines the resolution of the extracted image.
        :param image_type: the type of the extracted image: jpg or tif
        :return:
        """
        concentriq_image_info = self.get_image(image_id)  # get image info
        image_source = next(concentriq_image_info["data"]["imageData"]["imageSources"].values().__iter__())

        image_server_url = image_source["imageServerUrl"]
        image_server_json = self.session.get(image_server_url, verify=False)

        id_url = image_server_json["@id"]
        request_url = \
            f"{id_url}/{top_left_x},{top_left_y},{bottom_right_x},{bottom_right_y}/{width},/0/default.{image_type}"
        response = self.session.request(method="get", url=request_url, verify=False)  # extract patch
        return response.content
