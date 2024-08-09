import datetime
from datetime import datetime
from typing import Callable

import boto3
from botocore import UNSIGNED
from botocore.auth import S3SigV4Auth
from botocore.config import Config
from requests import Session


class Upload:

    def __init__(self, concentriq_endpoint: str, session: Session,
                 aws_access_key_id: str,
                 aws_region: str,
                 s3_endpoint: str,
                 s3_bucket_name: str):
        self.concentriq_endpoint = concentriq_endpoint
        self.session = session
        self.aws_access_key_id = aws_access_key_id
        self.aws_region = aws_region
        self.s3_endpoint = s3_endpoint
        self.s3_bucket_name = s3_bucket_name

    def sign_s3_multipart_url_for_image(self, payload, timestamp, canonical_request, concentriq_image_id):
        return self.session.get(
            f'{self.concentriq_endpoint}/auth/sign/s3-multipart-url/image/{concentriq_image_id}',
            params={
                'payload': payload,
                'nonce': timestamp,
                'canonicalRequest': canonical_request
            },
            verify=False)

    def _get_s3_client_for_uploading(self, concentriq_image_dict: dict):
        boto_session = boto3.Session(aws_access_key_id=self.aws_access_key_id, aws_secret_access_key='')
        credentials = boto_session.get_credentials()
        creds = credentials.get_frozen_credentials()

        def do_sign(request, **kwargs):
            auth_obj = S3SigV4Auth(creds, 's3', self.aws_region)
            timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            request.context['timestamp'] = timestamp
            auth_obj._modify_request_before_signing(request)
            canonical_request = auth_obj.canonical_request(request)
            payload = auth_obj.string_to_sign(request, canonical_request)
            response = self.sign_s3_multipart_url_for_image(
                payload, timestamp, canonical_request, concentriq_image_dict["id"])
            signature = response['data']['signature']
            auth_obj._inject_signature_to_request(request, signature)

        s3_client = boto3.client('s3', region_name=self.aws_region,
                                 endpoint_url=self.s3_endpoint,
                                 aws_access_key_id=self.aws_access_key_id,
                                 aws_secret_access_key='',  # leave this blank as a placeholder value
                                 config=Config(signature_version=UNSIGNED)
                                 )

        s3_client.meta.events.register_last('request-created.s3', do_sign)
        return s3_client

    def upload_image_by_s3_copy_object(self, source_image_key, source_s3_bucket, concentriq_image_dict):
        """
        Directly copy within AWS S3. ATTENTION: only works for files < 5GB
        """
        s3_client = self._get_s3_client_for_uploading(concentriq_image_dict)
        s3_client.copy_object(CopySource={"Bucket": source_s3_bucket, "Key": source_image_key},
                              Bucket=self.s3_bucket_name, Key=concentriq_image_dict['storageKey'])

    def upload_image_by_s3_copy(self, source_image_key, source_s3_bucket, source_client, concentriq_image_dict):
        s3_client = self._get_s3_client_for_uploading(concentriq_image_dict)
        s3_client.copy(CopySource={"Bucket": source_s3_bucket, "Key": source_image_key},
                              SourceClient=source_client,
                              Bucket=self.s3_bucket_name, Key=concentriq_image_dict['storageKey'])

    def upload_image_by_upload_fileobj(self, image_fileobj, concentriq_image_dict,
                                       callback: Callable[[int], None] = None):
        """
        Upload fileobj
        :param image_fileobj: the fileobj. NOTICE: not file path
        :param concentriq_image_dict:
        :param callback:
        :return:
        """
        s3_client = self._get_s3_client_for_uploading(concentriq_image_dict)
        s3_client.upload_fileobj(image_fileobj, self.s3_bucket_name, concentriq_image_dict['storageKey'],
                                 Callback=callback)
