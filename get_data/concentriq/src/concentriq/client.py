from concentriq.basis import ThreadSafeRequestsSession
from concentriq.endpoints import Annotations
from concentriq.endpoints import Folders
from concentriq.endpoints import ImageSets
from concentriq.endpoints import Images
from concentriq.endpoints import Metadata
from concentriq.endpoints import Upload


class Concentriq:
    def __init__(self,
                 email: str,
                 password: str,
                 endpoint: str,
                 aws_access_key_id: str = None,
                 aws_region: str = None,
                 s3_endpoint: str = None,
                 s3_bucket_name: str = None):
        self.endpoint = endpoint
        self.session = ThreadSafeRequestsSession(email, password)
        self.annotations = Annotations(endpoint, self.session)
        self.folders = Folders(endpoint, self.session)
        self.image_sets = ImageSets(endpoint, self.session)
        self.images = Images(endpoint, self.session)
        self.metadata = Metadata(endpoint, self.session)
        self.upload = Upload(endpoint, self.session, aws_access_key_id, aws_region, s3_endpoint, s3_bucket_name)

    def get_user(self):
        return self.session.get('{}/auth/user'.format(self.endpoint), verify=False)
