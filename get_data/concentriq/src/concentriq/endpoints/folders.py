import json
from collections import defaultdict

from concentriq.basis import ThreadSafeRequestsSession


class Folders:
    def __init__(self, concentriq_endpoint: str, session: ThreadSafeRequestsSession):
        self.endpoint = f"{concentriq_endpoint}/folders"
        self.session = session

    def create_folder(self, folder):
        return self.session.post(self.endpoint, json=folder, verify=False)

    def get_folders_by_image_set_id(self, image_set_id):
        res = []
        page = 1
        while True:
            params = {
                "filters": json.dumps({"imageSetId": [image_set_id]}),
                "pagination": json.dumps({"rowsPerPage": 1000, "sortBy": "name", "descending": False, "page": page})
            }
            result = self.session.get(self.endpoint, params=params, verify=False)
            data = result["data"]["folders"]
            pagination = result["meta"]["pagination"]
            res.extend(data)
            if pagination["rowsPerPage"] * page < pagination["totalRows"]:
                page += 1
            else:
                break

        return res

    def get_concentriq_folder_trie_by_image_set_id(self, image_set_id, parent_folder_id=None):
        """
        Get the image set's folders and construct them into a Trie:
        parentFolderName
        |
        V
        subfolder1Name,         subfolder2Name ...,  "*"-> parentFolder's Id
        |                           |
        V                           V
        subfolder1's sub        subfolder2's sub

        The Trie contains the folder Name. At each level, there is a key "*" whose value is the above level's folderId

        If parent_folder_id is given then the Trie is assembled from the parent_folder_id
        If the parent_folder_id is None then the Trie is assembled fully using None as root
        """
        res = {'*': parent_folder_id}
        folders = self.get_folders_by_image_set_id(image_set_id)
        folder_id_to_folder_name = dict()
        folder_to_sub_folder_ids = defaultdict(list)
        for folder in folders:
            folder_id_to_folder_name[folder["id"]] = folder["label"]
            folder_to_sub_folder_ids[folder["folderParentId"]].append(folder["id"])

        def recurse(root):
            current_folder_id = root['*']
            for sub_folder_id in folder_to_sub_folder_ids[current_folder_id]:
                sub_folder_name = folder_id_to_folder_name[sub_folder_id]
                if sub_folder_name in root:
                    sub_folder_name = sub_folder_name + "*"
                root[sub_folder_name] = dict()
                root[sub_folder_name]['*'] = sub_folder_id

                recurse(root[sub_folder_name])

        recurse(res)
        return res
